import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import platform

from data.cifar10 import get_cifar10, get_cifar10_separate
from models.ResNet import ResNet18
from torch.utils.data import TensorDataset, DataLoader

from detection import SimpleNetworksDetector, detect_poisoned_datasets, SampleLevelDetector, detect_poisoned_samples
from evaluation import evaluate_dataset, evaluate_model_on_dataset
from noise_generation import (
    generate_classwise_errmin_noise,
    generate_classwise_errmax_noise,
    generate_samplewise_errmin_noise,
    generate_samplewise_errmax_noise,
    generate_random_noise
)
from unlearnable_dataset import create_unlearnable_dataset, create_mixed_dataset, create_limited_training_loader
from train_utils import (
    train_model_with_noise, 
    train_model_on_mixed_dataset, 
    plot_training_curves,
    plot_ratio_impact
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_system_info():
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.release()}")
    if platform.system() == 'Linux':
        try:
            import subprocess
            cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode().strip()
            print(f"CPU: {cpu_info.split(':')[1].strip()}")
        except:
            pass
        
        try:
            memory_info = subprocess.check_output("free -h | head -2 | tail -1", shell=True).decode().strip()
            print(f"Memory: {memory_info.split()[1]}")
        except:
            pass
    
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            try:
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            except:
                pass
            
    try:
        print(f"CPU Count: {os.cpu_count()}")
        if 'NUMBA_NUM_THREADS' in os.environ:
            print(f"NUMBA threads: {os.environ['NUMBA_NUM_THREADS']}")
    except:
        pass
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Generate Unlearnable Examples & Evaluate")
    parser.add_argument("--type", type=str, 
                       choices=["classwise-min", "classwise-max", "samplewise-min", "samplewise-max", 
                                "random", "all-min", "all-max", "all"], 
                       required=True, help="Noise type to generate & test")
    parser.add_argument("--epsilon", type=float, default=8/255, 
                        help="Perturbation magnitude (L_inf)")
    parser.add_argument("--steps", type=int, default=20, 
                        help="Number of steps for generating noise")
    parser.add_argument("--alpha", type=float, default=2/255, 
                        help="Step size for noise generation")
    parser.add_argument("--train", action="store_true", 
                        help="If set, train models on the datasets and generate curves")
    parser.add_argument("--target-class", type=int, default=None,
                    help="Target class to poison (0-9 for CIFAR-10)")
    parser.add_argument("--train-epochs", type=int, default=30, 
                        help="Number of epochs for training models")
    parser.add_argument("--patience", type=int, default=15, 
                        help="Patience for early stopping")
    parser.add_argument("--test-mixed", action="store_true", 
                        help="If set, train models on mixed datasets with different ratios of unlearnable data")
    parser.add_argument("--ratios", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0", 
                        help="Comma-separated list of unlearnable data ratios to test")
    parser.add_argument("--model", type=str, default="ResNet18", choices=["ResNet18"], 
                        help="Model architecture to use for testing mixed datasets")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Number of samples per class for samplewise noise generation (optional)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers for data loading (default: auto)")
    parser.add_argument("--detect", action="store_true",
                        help="If set, run detection algorithm on datasets")
    parser.add_argument("--detection-bound", type=float, default=0.7,
                        help="Detection threshold for poisoned datasets")
    parser.add_argument("--detection-samples", type=int, default=10000,
                        help="Number of samples to use for detection")
    parser.add_argument("--detection-epochs", type=int, default=20,
                        help="Number of epochs for detection model training")
    parser.add_argument("--detect-samples", action="store_true",
                    help="If set, detect individual poisoned samples in test dataset")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                    help="Confidence threshold for sample detection")

    args = parser.parse_args()
    
    if args.num_workers is None:
        args.num_workers = min(4, os.cpu_count() or 1)
    
    epsilon = args.epsilon
    steps = args.steps
    alpha = args.alpha
    train_epochs = args.train_epochs
    patience = args.patience
    model_type = args.model
    samples_per_class = args.samples_per_class
    batch_size = args.batch_size
    ratios = [float(r) for r in args.ratios.split(',')]

    train_loader, test_loader = get_cifar10(
        batch_size=batch_size, 
        num_workers=args.num_workers
    )
    
    start_time = time.time()
    model = ResNet18(num_classes=10).to(device)
    print(f"Model on {device}: {next(model.parameters()).device}")
    
    if os.path.exists("resnet18_cifar10.pth"):
        model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))
        print("Loaded pretrained model weights from resnet18_cifar10.pth")
    else:
        print("Warning: pretrained resnet18_cifar10.pth not found, using random initialization.")
        print("Training a base model for initialization...")
        temp_train_loader, temp_test_loader = get_cifar10(batch_size=batch_size, num_workers=args.num_workers)
        base_model_results = train_model_with_noise(
            temp_train_loader, temp_test_loader,
            clean_train_loader=temp_train_loader,
            noise_type=None,
            epochs=5, patience=5
        )
        model = base_model_results['model']
        torch.save(model.state_dict(), "resnet18_cifar10.pth")
        print("Base model trained and saved.")

    print(f"\nEvaluating Unlearnable Examples on ResNet18/CIFAR-10:")
    print(f"Configuration: epsilon={epsilon}, steps={steps}, alpha={alpha}, samples_per_class={samples_per_class}")
    
    baseline_train_acc = evaluate_dataset(train_loader, model)
    baseline_test_acc = evaluate_dataset(test_loader, model)
    print(f"\nBaseline model - Train Acc: {baseline_train_acc:.2f}%, Test Acc: {baseline_test_acc:.2f}%")
    
    noise_collection = {}
    unlearnable_datasets = {}
    limited_train_loaders = {}
    mixed_datasets = {}
    
    if args.type in ["classwise-min", "all-min", "all"]:
        try:
            classwise_min_noise = generate_classwise_errmin_noise(train_loader, model, epsilon, steps, alpha)
            classwise_min_train_acc = evaluate_dataset(train_loader, model, classwise_min_noise, "classwise-min", apply_noise=True) 
            classwise_min_test_acc = evaluate_dataset(test_loader, model)
            
            print(f"Classwise Min-Min Noise - Train Acc (w/ noise): {classwise_min_train_acc:.2f}%, Test Acc: {classwise_min_test_acc:.2f}%")
            
            noise_collection["classwise-min"] = classwise_min_noise
            unlearnable_datasets["classwise-min"] = create_unlearnable_dataset(train_loader, classwise_min_noise, "classwise-min", epsilon)
            limited_train_loaders["classwise-min"] = create_limited_training_loader(
                train_loader, classwise_min_noise, "classwise-min", samples_per_class=samples_per_class
            )
        except Exception as e:
            print(f"Error generating classwise min-min noise: {e}")

    if args.type in ["classwise-max", "all-max", "all"]:
        try:
            classwise_max_noise = generate_classwise_errmax_noise(train_loader, model, epsilon, steps, alpha)
            classwise_max_train_acc = evaluate_dataset(train_loader, model, classwise_max_noise, "classwise-max", apply_noise=True)
        
            classwise_max_test_acc = evaluate_dataset(test_loader, model)
            
            print(f"Classwise Min-Max Noise (PGD) - Train Acc (w/ noise): {classwise_max_train_acc:.2f}%, Test Acc: {classwise_max_test_acc:.2f}%")
            
            noise_collection["classwise-max"] = classwise_max_noise
            unlearnable_datasets["classwise-max"] = create_unlearnable_dataset(train_loader, classwise_max_noise, "classwise-max", epsilon)
            limited_train_loaders["classwise-max"] = create_limited_training_loader(
                train_loader, classwise_max_noise, "classwise-max", samples_per_class=samples_per_class
            )
        except Exception as e:
            print(f"Error generating classwise min-max noise: {e}")

    if args.type in ["samplewise-min", "all-min", "all"]:
        try:
            samplewise_min_noise = generate_samplewise_errmin_noise(
                train_loader, model, epsilon, steps, alpha, samples_per_class=samples_per_class
            )

            if 'images' in samplewise_min_noise and 'labels' in samplewise_min_noise:
                temp_dataset = torch.utils.data.TensorDataset(
                    samplewise_min_noise['images'] + samplewise_min_noise['noises'],
                    samplewise_min_noise['labels']
                )
                temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
                
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in temp_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                samplewise_min_train_acc = 100 * correct / total
            else:
                samplewise_min_train_acc = evaluate_dataset(train_loader, model, samplewise_min_noise, "samplewise-min", apply_noise=True)
            
            samplewise_min_test_acc = evaluate_dataset(test_loader, model)
            
            print(f"Samplewise Min-Min Noise - Train Acc (w/ noise): {samplewise_min_train_acc:.2f}%, Test Acc: {samplewise_min_test_acc:.2f}%")
            
            noise_collection["samplewise-min"] = samplewise_min_noise
            unlearnable_datasets["samplewise-min"] = create_unlearnable_dataset(train_loader, samplewise_min_noise, "samplewise-min", epsilon)
            limited_train_loaders["samplewise-min"] = create_limited_training_loader(
                train_loader, samplewise_min_noise, "samplewise-min", samples_per_class=samples_per_class
            )
        except Exception as e:
            print(f"Error generating samplewise min-min noise: {e}")

    if args.type in ["samplewise-max", "all-max", "all"]:
        try:
            samplewise_max_noise = generate_samplewise_errmax_noise(
                train_loader, model, epsilon, steps, alpha, samples_per_class=samples_per_class
            )
            if 'images' in samplewise_max_noise and 'labels' in samplewise_max_noise:
                temp_dataset = torch.utils.data.TensorDataset(
                    samplewise_max_noise['images'] + samplewise_max_noise['noises'],
                    samplewise_max_noise['labels']
                )
                temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
                model.train()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in temp_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                samplewise_max_train_acc = 100 * correct / total
            else:
                samplewise_max_train_acc = evaluate_dataset(train_loader, model, samplewise_max_noise, "samplewise-max", apply_noise=True)
            samplewise_max_test_acc = evaluate_dataset(test_loader, model)   
            print(f"Samplewise Min-Max Noise (PGD) - Train Acc (w/ noise): {samplewise_max_train_acc:.2f}%, Test Acc: {samplewise_max_test_acc:.2f}%")
            
            noise_collection["samplewise-max"] = samplewise_max_noise
            unlearnable_datasets["samplewise-max"] = create_unlearnable_dataset(train_loader, samplewise_max_noise, "samplewise-max", epsilon)
            limited_train_loaders["samplewise-max"] = create_limited_training_loader(
                train_loader, samplewise_max_noise, "samplewise-max", samples_per_class=samples_per_class
            )
        except Exception as e:
            print(f"Error generating samplewise min-max noise: {e}")

    if args.type in ["random", "all-min", "all-max", "all"]:
        try:
            random_noise = generate_random_noise((10, 3, 32, 32), epsilon)
            random_train_acc = evaluate_dataset(train_loader, model, random_noise, "random", apply_noise=True)
            random_test_acc = evaluate_dataset(test_loader, model)
            
            print(f"Random Noise - Train Acc (w/ noise): {random_train_acc:.2f}%, Test Acc: {random_test_acc:.2f}%")
            
            noise_collection["random"] = random_noise
            unlearnable_datasets["random"] = create_unlearnable_dataset(train_loader, random_noise, "random", epsilon)
            limited_train_loaders["random"] = create_limited_training_loader(
                train_loader, random_noise, "random", samples_per_class=samples_per_class
            )
        except Exception as e:
            print(f"Error generating random noise: {e}")

    if args.train:
        print("STARTING MODEL TRAINING PHASE")
        
        training_results = []

        print("\nTraining clean model from scratch:")
        try:
            clean_results = train_model_with_noise(
                train_loader, test_loader, clean_train_loader=train_loader,
                noise_type=None, epochs=train_epochs, patience=patience
            )
            training_results.append(clean_results)
        except Exception as e:
            print(f"Error training clean model: {e}")
        
        for noise_type, limited_loader in limited_train_loaders.items():
            print("\n" + "-"*50)
            print(f"Training new model with {noise_type} noise (limited dataset):")
            print("-"*50)
            
            new_model = ResNet18(num_classes=10).to(device)
            
            try:
                noise_results = train_model_with_noise(
                    limited_loader, test_loader, clean_train_loader=train_loader,
                    noise_type=noise_type, epochs=train_epochs, patience=patience
                )
                training_results.append(noise_results)
            except Exception as e:
                print(f"Error training model with {noise_type} noise: {e}")
        
        if training_results:
            print("\nGenerating comparison plots...")
            try:
                plot_training_curves(training_results)
                print("Training complete. Results saved to results directory.")
            except Exception as e:
                print(f"Error plotting training curves: {e}")
        
    if args.test_mixed and unlearnable_datasets:
        print("\n" + "="*50)
        print("TESTING DIFFERENT RATIOS OF UNLEARNABLE DATA")
        print("="*50)
        
        for noise_type, unlearnable_dataset in unlearnable_datasets.items():
            print(f"\nTesting impact of {noise_type} unlearnable data ratio:")
            ratio_results = []
            
            for ratio in ratios:
                print(f"\nTraining with {ratio:.2f} ratio of {noise_type} unlearnable data:")
                try:
                    mixed_dataset = create_mixed_dataset(train_loader, unlearnable_dataset, ratio,target_class=args.target_class)
                    mixed_datasets[f"{noise_type}_{ratio:.2f}"] = mixed_dataset
                    
                    results = train_model_on_mixed_dataset(
                        mixed_dataset, clean_train_loader=train_loader, test_loader=test_loader,
                        model_type=model_type, 
                        epochs=train_epochs,
                        batch_size=batch_size,
                        patience=patience
                    )
                    ratio_results.append(results)
                except Exception as e:
                    print(f"Error training with {ratio:.2f} ratio of {noise_type} noise: {e}")
            
            if ratio_results:
                try:
                    plot_ratio_impact(ratio_results, noise_type)
                    plot_training_curves(ratio_results)
                except Exception as e:
                    print(f"Error plotting ratio results: {e}")
    
    if args.detect and unlearnable_datasets:
        print("\n" + "="*50)
        print("RUNNING DETECTION ON DATASETS")
        print("="*50)
        
        detector_params = {
            "model_type": model_type,
            "detection_bound": args.detection_bound,
            "sample_size": args.detection_samples,
            "epochs": args.detection_epochs,
            "batch_size": batch_size,
            "verbose": True
        }
        
        detector = SimpleNetworksDetector(**detector_params)
        
        print("\nDetecting on clean dataset:")
        clean_detection = detector.detect(train_loader, test_loader)
        detector.visualize_results(clean_detection, "results/detection/clean_dataset.png")
        
        detection_results = {"clean": clean_detection}
        
        for noise_type, unlearnable_dataset in unlearnable_datasets.items():
            print(f"\nDetecting on {noise_type} unlearnable dataset:")
            try:
                result = detector.detect(unlearnable_dataset, test_loader)
                detection_results[noise_type] = result
                detector.visualize_results(result, f"results/detection/{noise_type}_dataset.png")
            except Exception as e:
                print(f"Error during detection on {noise_type} dataset: {e}")
        
        if mixed_datasets:
            for name, mixed_dataset in mixed_datasets.items():
                print(f"\nDetecting on mixed dataset: {name}")
                try:
                    result = detector.detect(mixed_dataset, test_loader)
                    detection_results[name] = result
                    detector.visualize_results(result, f"results/detection/mixed_{name}.png")
                except Exception as e:
                    print(f"Error during detection on mixed dataset {name}: {e}")
        
        try:
            plt.figure(figsize=(12, 6))
            
            names = list(detection_results.keys())
            scores = [detection_results[name]['detection_score'] for name in names]
            is_poisoned = [detection_results[name]['is_poisoned'] for name in names]
            
            colors = ['red' if poisoned else 'green' for poisoned in is_poisoned]
            
            plt.bar(names, scores, color=colors)
            plt.axhline(y=detector.detection_bound, color='k', linestyle='--', 
                       label=f'Detection Threshold ({detector.detection_bound:.2f})')
            
            plt.title('Detection Scores Comparison')
            plt.xlabel('Dataset')
            plt.ylabel('Detection Score')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            os.makedirs("results/detection", exist_ok=True)
            plt.savefig("results/detection/detection_comparison.png", dpi=300)
            print(f"Detection comparison saved to results/detection/detection_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Error creating detection comparison plot: {e}")

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nExperiment complete! Total runtime: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    
    
    if args.detect_samples:
        print("\n" + "="*50)
        print("RUNNING SAMPLE-LEVEL DETECTION")
        print("="*50)

        clean_train_loader, clean_test_loader = get_cifar10(
            batch_size=batch_size, 
            num_workers=args.num_workers
        )

        sample_detector_params = {
            "model_type": model_type,
            "confidence_threshold": args.confidence_threshold,
            "epochs": 20,
            "batch_size": batch_size,
            "verbose": True
        }

        print("\nDetecting difficult samples in clean test dataset:")
        clean_results = detect_poisoned_samples(
            clean_test_loader, 
            clean_train_loader,
            detector_params=sample_detector_params,
            output_dir="results/sample_detection/clean"
        )

        for noise_type, unlearnable_dataset in unlearnable_datasets.items():
            print(f"\nDetecting poisoned samples in {noise_type} unlearnable dataset:")

            if isinstance(unlearnable_dataset, dict) and 'images' in unlearnable_dataset and 'labels' in unlearnable_dataset:
                unlearnable_test_dataset = TensorDataset(
                    unlearnable_dataset['images'].cpu(),
                    unlearnable_dataset['labels'].cpu()
                )
                unlearnable_test_loader = DataLoader(
                    unlearnable_test_dataset, 
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )

                try:
                    unlearnable_results = detect_poisoned_samples(
                        unlearnable_test_loader,
                        clean_train_loader,
                        detector_params=sample_detector_params,
                        output_dir=f"results/sample_detection/{noise_type}"
                    )

                    clean_suspicious_rate = clean_results['suspicious_samples'] / clean_results['total_samples']
                    unlearnable_suspicious_rate = unlearnable_results['suspicious_samples'] / unlearnable_results['total_samples']

                    print(f"\nComparison of suspicious sample rates:")
                    print(f"Clean test data: {clean_suspicious_rate*100:.2f}%")
                    print(f"{noise_type} data: {unlearnable_suspicious_rate*100:.2f}%")
                    print(f"Ratio: {unlearnable_suspicious_rate/clean_suspicious_rate:.2f}x more suspicious samples")

                    plt.figure(figsize=(10, 6))
                    plt.bar(['Clean', noise_type], [clean_suspicious_rate*100, unlearnable_suspicious_rate*100])
                    plt.title(f'Suspicious Sample Rate Comparison')
                    plt.ylabel('Suspicious Samples (%)')
                    plt.grid(True, linestyle='--', alpha=0.7)

                    comparison_path = f"results/sample_detection/comparison_{noise_type}.png"
                    plt.savefig(comparison_path, dpi=300)
                    plt.close()
                    print(f"Comparison saved to {comparison_path}")

                except Exception as e:
                    print(f"Error during sample detection on {noise_type} dataset: {e}")
            else:
                print(f"Cannot run sample detection on {noise_type} dataset: incompatible format")

if __name__ == "__main__":
    main()