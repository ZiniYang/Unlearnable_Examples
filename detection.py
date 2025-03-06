import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from models.ResNet import ResNet18
from data.cifar10 import get_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256, num_classes=10):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleNetworksDetector:
    def __init__(self, 
                 model_type="SimpleNN", 
                 detection_bound=0.7, 
                 sample_size=10000, 
                 epochs=20, 
                 batch_size=128,
                 random_seed=42,
                 verbose=True):
        self.model_type = model_type
        self.detection_bound = detection_bound
        self.sample_size = sample_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.verbose = verbose
        self.model = None
        
    def create_model(self):
        if self.model_type == "SimpleNN":
            return SimpleNN(input_dim=3072, hidden_dim=256, num_classes=10).to(device)
        elif self.model_type == "ResNet18":
            return ResNet18(num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def detect(self, dataset, clean_test_loader=None):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        start_time = time.time()

        if isinstance(dataset, DataLoader):
            all_images = []
            all_labels = []
            for images, labels in dataset:
                all_images.append(images)
                all_labels.append(labels)

            all_images = torch.cat(all_images)
            all_labels = torch.cat(all_labels)
            full_dataset = TensorDataset(all_images, all_labels)
        elif isinstance(dataset, dict) and 'images' in dataset and 'labels' in dataset:
            images = dataset['images'].cpu()
            labels = dataset['labels'].cpu()
            full_dataset = TensorDataset(images, labels)
        elif isinstance(dataset, TensorDataset):
            full_dataset = dataset
        else:
            raise ValueError("Unsupported dataset type")

        dataset_size = len(full_dataset)
        sample_size = min(self.sample_size, dataset_size)

        indices = torch.randperm(dataset_size)[:sample_size]
        detection_dataset = Subset(full_dataset, indices)

        train_size = int(0.8 * sample_size)
        val_size = sample_size - train_size

        train_dataset, val_dataset = random_split(detection_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=0, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        model = self.create_model()
        self.model = model
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        if self.verbose:
            print(f"Training detection model ({self.model_type}) on {train_size} samples for {self.epochs} epochs")

        try:
            for epoch in range(self.epochs):
                model.train()
                train_correct = 0
                train_total = 0
                train_loss = 0.0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_loss += loss.item() * images.size(0)

                train_acc = 100.0 * train_correct / train_total
                train_loss = train_loss / train_total

                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        val_loss += loss.item() * images.size(0)

                val_acc = 100.0 * val_correct / val_total
                val_loss = val_loss / val_total

                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)

                if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                    print(f"Epoch {epoch+1}/{self.epochs} | "
                          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                scheduler.step()

        except Exception as e:
            print(f"Error during detection training: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(history['val_acc']) > 0:
                val_acc = history['val_acc'][-1]
                train_acc = history['train_acc'][-1]
                is_poisoned = val_acc <= self.detection_bound * 100

                results = {
                    'is_poisoned': is_poisoned,
                    'detection_score': val_acc / 100,
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'test_accuracy': None,
                    'history': history,
                    'detection_time': time.time() - start_time,
                    'sample_size': sample_size,
                    'detection_bound': self.detection_bound,
                    'error': str(e)
                }

                return results
            else:
                raise

        test_acc = None
        if clean_test_loader:
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, labels in clean_test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_acc = 100.0 * test_correct / test_total
            if self.verbose:
                print(f"Test Accuracy on clean data: {test_acc:.2f}%")

        is_poisoned = val_acc <= self.detection_bound * 100

        end_time = time.time()
        detection_time = end_time - start_time

        results = {
            'is_poisoned': is_poisoned,
            'detection_score': val_acc / 100,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'history': history,
            'detection_time': detection_time,
            'sample_size': sample_size,
            'detection_bound': self.detection_bound
        }

        if self.verbose:
            print(f"\nDetection Results:")
            print(f"Dataset {'IS' if is_poisoned else 'is NOT'} poisoned")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            if test_acc is not None:
                print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Detection Threshold: {self.detection_bound * 100:.2f}%")
            print(f"Detection Time: {detection_time:.2f} seconds")
        
        suspicious_results = self.evaluate_suspicious_samples(dataset)
        results['suspicious_rate'] = suspicious_results['suspicious_rate']
        results['suspicious_count'] = suspicious_results['suspicious_count']
        results['accuracy'] = suspicious_results['accuracy']
        
        if self.verbose:
            print(f"Detection Accuracy: {suspicious_results['accuracy']:.2f}%")
            print(f"Suspicious Samples: {suspicious_results['suspicious_count']} ({suspicious_results['suspicious_rate']:.2f}%)")

        return results
    
    def evaluate_suspicious_samples(self, dataset):
        if self.model is None:
            raise ValueError("No model available. Run detect() first.")
            
        if not isinstance(dataset, DataLoader):
            if isinstance(dataset, dict) and 'images' in dataset and 'labels' in dataset:
                dataset = TensorDataset(dataset['images'].cpu(), dataset['labels'].cpu())
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        else:
            data_loader = dataset
            
        self.model.eval()
        
        confidences = []
        predictions = []
        correct = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                conf, pred = torch.max(probs, dim=1)
                
                confidences.extend(conf.cpu().numpy())
                predictions.extend(pred.cpu().numpy())
                correct.extend((pred == labels).cpu().numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        correct = np.array(correct)
        
        threshold = 0.8
        suspicious_mask = confidences < threshold
        suspicious_count = np.sum(suspicious_mask)
        
        accuracy = np.mean(correct) * 100
        avg_confidence = np.mean(confidences)
        suspicious_rate = (suspicious_count / len(confidences)) * 100
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'suspicious_count': suspicious_count,
            'suspicious_rate': suspicious_rate,
            'confidences': confidences,
            'predictions': predictions,
            'correct': correct
        }
        
    def visualize_results(self, results, save_path=None):
        history = results['history']
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(history['val_acc'], 'r-', label='Validation Accuracy')
        
        if results['test_accuracy'] is not None:
            plt.axhline(y=results['test_accuracy'], color='g', linestyle='--', 
                        label='Clean Test Accuracy')
        
        plt.axhline(y=results['detection_bound'] * 100, color='k', linestyle=':',
                   label=f'Detection Threshold ({results["detection_bound"] * 100:.1f}%)')
        
        plt.title(f"Detection Results: Dataset {'IS' if results['is_poisoned'] else 'is NOT'} Poisoned")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(2, 1, 2)
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Detection visualization saved to {save_path}")
        
        plt.close()

def detect_poisoned_datasets(dataset_paths, detector_params=None, clean_test_loader=None, output_dir="results/detection"):
    if detector_params is None:
        detector_params = {}
    
    if 'model_type' not in detector_params:
        detector_params['model_type'] = "SimpleNN"
    
    detector = SimpleNetworksDetector(**detector_params)
    
    results = {}
    datasets = {}
    
    for path_or_obj in dataset_paths:
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                print(f"Warning: Dataset path {path_or_obj} does not exist. Skipping.")
                continue
            
            dataset_name = os.path.basename(path_or_obj).split('.')[0]
            dataset = torch.load(path_or_obj)
            datasets[dataset_name] = dataset
        else:
            if hasattr(path_or_obj, 'name'):
                dataset_name = path_or_obj.name
            else:
                dataset_name = f"dataset_{len(datasets)}"
            datasets[dataset_name] = path_or_obj
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, dataset in datasets.items():
        print(f"\nRunning detection on dataset: {name}")
        result = detector.detect(dataset, clean_test_loader)
        results[name] = result
        
        save_path = os.path.join(output_dir, f"detection_{name}.png")
        detector.visualize_results(result, save_path)
    
    if len(results) > 1:
        try:
            plt.figure(figsize=(15, 15))
            
            plt.subplot(3, 1, 1)
            names = list(results.keys())
            scores = [results[name]['detection_score'] for name in names]
            is_poisoned = [results[name]['is_poisoned'] for name in names]
            
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
            
            plt.subplot(3, 1, 2)
            train_accs = [results[name]['train_accuracy'] for name in names]
            val_accs = [results[name]['val_accuracy'] for name in names]
            test_accs = [results[name]['test_accuracy'] if results[name]['test_accuracy'] is not None else 0 for name in names]
            
            x = np.arange(len(names))
            width = 0.25
            
            plt.bar(x - width, train_accs, width, label='Training Accuracy')
            plt.bar(x, val_accs, width, label='Validation Accuracy')
            if any(test_accs):
                plt.bar(x + width, test_accs, width, label='Test Accuracy')
            
            plt.title('Accuracy Comparison')
            plt.xlabel('Dataset')
            plt.ylabel('Accuracy (%)')
            plt.xticks(x, names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(3, 1, 3)
            suspicious_rates = [results[name]['suspicious_rate'] for name in names]
            
            plt.bar(names, suspicious_rates, color=colors)
            plt.title('Suspicious Sample Rate Comparison')
            plt.xlabel('Dataset')
            plt.ylabel('Suspicious Sample Rate (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            comparison_path = os.path.join(output_dir, "detection_comparison.png")
            plt.savefig(comparison_path, dpi=300)
            print(f"Detection comparison saved to {comparison_path}")
            plt.close()
        except Exception as e:
            print(f"Error creating detection comparison plot: {e}")
    
    return results

class SampleLevelDetector:
    def __init__(self, model_type="SimpleNN", confidence_threshold=0.8, epochs=20, batch_size=128, verbose=True):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
    def create_model(self):
        if self.model_type == "SimpleNN":
            return SimpleNN(input_dim=3072, hidden_dim=256, num_classes=10).to(device)
        elif self.model_type == "ResNet18":
            return ResNet18(num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def train_model(self, train_loader):
        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        if self.verbose:
            print(f"Training detection model ({self.model_type}) for {self.epochs} epochs")
            
        for epoch in range(self.epochs):
            model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()
            
            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                train_acc = 100.0 * train_correct / train_total
                print(f"Epoch {epoch+1}/{self.epochs} | Train Acc: {train_acc:.2f}% | Loss: {train_loss/len(train_loader):.4f}")
            
            scheduler.step()
        
        return model
    
    def detect_samples(self, test_loader, model):
        model.eval()
        all_confidences = []
        all_predictions = []
        all_labels = []
        all_correct = []
        suspicious_indices = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                for i in range(batch_size):
                    if conf[i] < self.confidence_threshold:
                        suspicious_indices.append(total_samples + i)
                
                all_confidences.extend(conf.cpu().numpy())
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_correct.extend((pred == labels).cpu().numpy())
                total_samples += batch_size
        
        accuracy = 100.0 * np.mean(all_correct)
        avg_confidence = np.mean(all_confidences)
        suspicious_count = len(suspicious_indices)
        
        results = {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'total_samples': total_samples,
            'suspicious_samples': suspicious_count,
            'suspicious_indices': suspicious_indices,
            'all_confidences': all_confidences,
            'all_predictions': all_predictions,
            'all_labels': all_labels
        }
        
        if self.verbose:
            print(f"\nSample Detection Results:")
            print(f"Overall Accuracy: {accuracy:.2f}%")
            print(f"Average Confidence: {avg_confidence:.4f}")
            print(f"Suspicious Samples: {suspicious_count} out of {total_samples} ({100.0 * suspicious_count / total_samples:.2f}%)")
        
        return results

def detect_poisoned_samples(test_loader, clean_train_loader, detector_params=None, output_dir="results/sample_detection"):
    if detector_params is None:
        detector_params = {}
    
    if 'model_type' not in detector_params:
        detector_params['model_type'] = "SimpleNN"
        
    detector = SampleLevelDetector(**detector_params)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = detector.train_model(clean_train_loader)
    
    results = detector.detect_samples(test_loader, model)
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(results['all_confidences'], bins=30)
        plt.axvline(x=detector.confidence_threshold, color='r', linestyle='--', 
                  label=f'Threshold ({detector.confidence_threshold:.2f})')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=300)
        plt.close()
        
        np.save(os.path.join(output_dir, "suspicious_indices.npy"), np.array(results['suspicious_indices']))
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return results