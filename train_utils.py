import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time

from evaluation import evaluate_dataset, evaluate_model_on_dataset
from models.ResNet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model_with_noise(train_loader, test_loader, clean_train_loader=None, noise_type=None, epochs=100, patience=10):
    model = ResNet18(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'results/model_{noise_type if noise_type else "clean"}_best.pth')
    
    train_accuracies = []  
    clean_train_accuracies = [] 
    test_accuracies = [] 
    epochs_list = []

    os.makedirs('results', exist_ok=True)
    
    print(f"\nTraining model with {noise_type if noise_type else 'clean'} data:")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item() * images.size(0)
        
        epoch_loss = epoch_loss / total

        train_acc = 100 * correct / total
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            test_acc = evaluate_dataset(test_loader, model)
          
            if clean_train_loader:
                clean_train_acc = evaluate_dataset(clean_train_loader, model)
            else:
                clean_train_acc = train_acc
            
            train_accuracies.append(train_acc)
            clean_train_accuracies.append(clean_train_acc)
            test_accuracies.append(test_acc)
            epochs_list.append(epoch + 1)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Acc (w/ noise): {train_acc:.2f}% | "
                  f"Clean Train Acc: {clean_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {epoch_loss:.4f}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(torch.load(f'results/model_{noise_type if noise_type else "clean"}_best.pth'))
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Acc (w/ noise): {train_acc:.2f}% | Loss: {epoch_loss:.4f}")
        
        scheduler.step()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    model_filename = f"results/model_{noise_type if noise_type else 'clean'}.pth"
    torch.save(model.state_dict(), model_filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_accuracies, 'b-', label='Train Accuracy (w/ noise)')
    plt.plot(epochs_list, clean_train_accuracies, 'r-', label='Clean Train Accuracy')
    plt.plot(epochs_list, test_accuracies, 'g-', label='Test Accuracy')
    plt.title(f'Model Accuracy with {noise_type if noise_type else "clean"} data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/accuracy_{noise_type if noise_type else "clean"}.png')
    plt.close()
    
    return {
        'epochs': epochs_list,
        'train_acc': train_accuracies,
        'clean_train_acc': clean_train_accuracies,
        'test_acc': test_accuracies,
        'noise_type': noise_type,
        'training_time': training_time,
        'model': model
    }

def train_model_on_mixed_dataset(mixed_dataset, clean_train_loader=None, test_loader=None, model_type="ResNet18", epochs=100, batch_size=128, patience=10):
    images = mixed_dataset['images'].cpu()
    labels = mixed_dataset['labels'].cpu()
    
    dataset = TensorDataset(images, labels)
    
    num_workers = 0
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)
    
    ratio = mixed_dataset.get('unlearnable_ratio', 0.0)
    noise_type = mixed_dataset.get('noise_type', 'unknown')
    
    print("\n============ Dataset Debug Info ============")
    print(f"Mixing ratio: {ratio:.2f}")
    print(f"Noise type: {noise_type}")
    
    if clean_train_loader is not None:
        try:
            mixed_batch = next(iter(loader))
            clean_batch = next(iter(clean_train_loader))
            
            mixed_images, mixed_labels = mixed_batch
            clean_images, clean_labels = clean_batch
            
            print(f"Mixed data batch shape: {mixed_images.shape}, labels shape: {mixed_labels.shape}")
            print(f"Clean data batch shape: {clean_images.shape}, labels shape: {clean_labels.shape}")
            
            print(f"Mixed data range: [{mixed_images.min().item():.4f}, {mixed_images.max().item():.4f}]")
            print(f"Clean data range: [{clean_images.min().item():.4f}, {clean_images.max().item():.4f}]")
            
            if ratio == 0.0:
                print("\n=== Detailed check for ratio 0 ===")
                
                temp_model = ResNet18(num_classes=10).to(device)
                temp_model.eval()
                
                with torch.no_grad():
                    mixed_images_gpu = mixed_images.to(device)
                    mixed_labels_gpu = mixed_labels.to(device)
                    mixed_outputs = temp_model(mixed_images_gpu)
                    _, mixed_preds = torch.max(mixed_outputs.data, 1)
                    mixed_acc = 100 * (mixed_preds == mixed_labels_gpu).float().mean().item()
                    
                    clean_images_gpu = clean_images.to(device)
                    clean_labels_gpu = clean_labels.to(device)
                    clean_outputs = temp_model(clean_images_gpu)
                    _, clean_preds = torch.max(clean_outputs.data, 1)
                    clean_acc = 100 * (clean_preds == clean_labels_gpu).float().mean().item()
                    
                    print(f"First batch - Mixed data accuracy: {mixed_acc:.2f}%, Clean data accuracy: {clean_acc:.2f}%")
                
                print("Checking complete dataset (please wait)...")
                
                mixed_eval_acc = evaluate_dataset(loader, temp_model)
                clean_eval_acc = evaluate_dataset(clean_train_loader, temp_model)
                
                print(f"Complete dataset - Mixed data accuracy: {mixed_eval_acc:.2f}%, Clean data accuracy: {clean_eval_acc:.2f}%")
                
                print("\nChecking data loader contents...")
                
                all_mixed_data = []
                all_clean_data = []
                
                for batch_idx, (m_imgs, m_lbls) in enumerate(loader):
                    if batch_idx < 3:
                        all_mixed_data.append((m_imgs, m_lbls))
                
                for batch_idx, (c_imgs, c_lbls) in enumerate(clean_train_loader):
                    if batch_idx < 3:
                        all_clean_data.append((c_imgs, c_lbls))
                
                for i in range(min(len(all_mixed_data), len(all_clean_data))):
                    m_imgs, m_lbls = all_mixed_data[i]
                    c_imgs, c_lbls = all_clean_data[i]
                    
                    print(f"Batch {i} - Shape match: {m_imgs.shape == c_imgs.shape}, {m_lbls.shape == c_lbls.shape}")
                    
                    if m_imgs.shape == c_imgs.shape:
                        img_diff = (m_imgs - c_imgs).abs().mean().item()
                        print(f"Batch {i} - Image mean absolute difference: {img_diff:.6f}")
                        
                        if m_lbls.shape == c_lbls.shape:
                            label_match = (m_lbls == c_lbls).float().mean().item()
                            print(f"Batch {i} - Label match rate: {label_match:.4f}")
                
                print("\nTest data statistics:")
                
                mixed_labels_list = []
                for _, lbls in loader:
                    mixed_labels_list.append(lbls)
                if mixed_labels_list:
                    all_mixed_labels = torch.cat(mixed_labels_list)
                    unique_mixed, counts_mixed = torch.unique(all_mixed_labels, return_counts=True)
                    print("Mixed dataset label distribution:")
                    for lbl, cnt in zip(unique_mixed.tolist(), counts_mixed.tolist()):
                        print(f"  Label {lbl}: {cnt} samples")
                
                clean_labels_list = []
                for _, lbls in clean_train_loader:
                    clean_labels_list.append(lbls)
                if clean_labels_list:
                    all_clean_labels = torch.cat(clean_labels_list)
                    unique_clean, counts_clean = torch.unique(all_clean_labels, return_counts=True)
                    print("Clean dataset label distribution:")
                    for lbl, cnt in zip(unique_clean.tolist(), counts_clean.tolist()):
                        print(f"  Label {lbl}: {cnt} samples")
                
                print("=== Detailed check end ===\n")
        except Exception as e:
            print(f"Error during debugging: {e}")
    
    print("============ Debug Info End ============\n")
    
    if model_type == "ResNet18":
        model = ResNet18(num_classes=10).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, 
                                  path=f'results/model_mixed_{noise_type}_{ratio:.2f}_best.pth')
    
    train_accuracies = []
    clean_train_accuracies = []
    test_accuracies = []
    epochs_list = []
    
    print(f"\nTraining model on mixed dataset with {ratio:.2f} ratio of {noise_type} unlearnable data:")
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0
            epoch_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item() * images.size(0)
            
            epoch_loss = epoch_loss / total
            
            train_acc_training = 100 * correct / total

            if epoch % 5 == 0 or epoch == epochs - 1:
                was_training = model.training
                
                model.eval()
                
                if epoch % 10 == 0 and ratio == 0.0:
                    print("\n======== Evaluation Phase Debug ========")
                    mixed_sample_acc = 0
                    mixed_counts = 0
                    with torch.no_grad():
                        for mixed_imgs, mixed_lbls in loader:
                            if mixed_counts < 5:
                                mixed_imgs, mixed_lbls = mixed_imgs.to(device), mixed_lbls.to(device)
                                mixed_outs = model(mixed_imgs)
                                _, mixed_preds = torch.max(mixed_outs.data, 1)
                                mixed_sample_acc = 100 * (mixed_preds == mixed_lbls).float().mean().item()
                                print(f"Mixed batch {mixed_counts} - Accuracy: {mixed_sample_acc:.2f}%")
                                mixed_counts += 1
                    
                    clean_sample_acc = 0
                    clean_counts = 0
                    with torch.no_grad():
                        for clean_imgs, clean_lbls in clean_train_loader:
                            if clean_counts < 5:
                                clean_imgs, clean_lbls = clean_imgs.to(device), clean_lbls.to(device)
                                clean_outs = model(clean_imgs)
                                _, clean_preds = torch.max(clean_outs.data, 1)
                                clean_sample_acc = 100 * (clean_preds == clean_lbls).float().mean().item()
                                print(f"Clean batch {clean_counts} - Accuracy: {clean_sample_acc:.2f}%")
                                clean_counts += 1
                    
                    print("======== Evaluation Debug End ========\n")
                
                mixed_eval_acc = evaluate_dataset(loader, model)
                
                if clean_train_loader:
                    clean_train_acc = evaluate_dataset(clean_train_loader, model)
                else:
                    clean_train_acc = None
                
                if test_loader:
                    test_acc = evaluate_dataset(test_loader, model)
                else:
                    test_acc = evaluate_model_on_dataset(model)
                
                if was_training:
                    model.train()
                
                train_acc = mixed_eval_acc
                
                train_accuracies.append(train_acc)
                if clean_train_acc is not None:
                    clean_train_accuracies.append(clean_train_acc)
                test_accuracies.append(test_acc)
                epochs_list.append(epoch + 1)
                
                if clean_train_acc is not None:
                    print(f"Epoch {epoch+1}/{epochs} | Train Acc (mixed, train mode): {train_acc_training:.2f}% | "
                        f"Train Acc (mixed, eval mode): {train_acc:.2f}% | "
                        f"Clean Train Acc: {clean_train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {epoch_loss:.4f}")
                    print(f"Gap: {abs(train_acc - clean_train_acc):.2f}%")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | Train Acc (mixed, train mode): {train_acc_training:.2f}% | "
                        f"Train Acc (mixed, eval mode): {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}% | Loss: {epoch_loss:.4f}")
                    
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(torch.load(f'results/model_mixed_{noise_type}_{ratio:.2f}_best.pth'))
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Acc (mixed, train mode): {train_acc_training:.2f}% | Loss: {epoch_loss:.4f}")
            
            scheduler.step()
    except Exception as e:
        print(f"Error during training: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(train_accuracies) > 0:
            result = {
                'epochs': epochs_list,
                'train_acc': train_accuracies,
                'test_acc': test_accuracies,
                'noise_type': noise_type,
                'unlearnable_ratio': ratio,
                'training_time': time.time() - start_time,
                'model': model
            }
            if clean_train_accuracies:
                result['clean_train_acc'] = clean_train_accuracies
            return result
        else:
            raise
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    os.makedirs('results', exist_ok=True)
    model_filename = f"results/model_mixed_{noise_type}_{ratio:.2f}.pth"
    torch.save(model.state_dict(), model_filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_accuracies, 'b-', label='Mixed Training Accuracy')
    if clean_train_accuracies:
        plt.plot(epochs_list, clean_train_accuracies, 'r-', label='Clean Train Accuracy')
    plt.plot(epochs_list, test_accuracies, 'g-', label='Test Accuracy')
    plt.title(f'Model Accuracy with {ratio:.2f} ratio of {noise_type} unlearnable data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/accuracy_mixed_{noise_type}_{ratio:.2f}.png')
    plt.close()
    
    result = {
        'epochs': epochs_list,
        'train_acc': train_accuracies,
        'test_acc': test_accuracies,
        'noise_type': noise_type,
        'unlearnable_ratio': ratio,
        'training_time': training_time,
        'model': model
    }
    
    if clean_train_accuracies:
        result['clean_train_acc'] = clean_train_accuracies
        
    return result

def plot_training_curves(training_results_list):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    for results in training_results_list:
        noise_type = results.get('noise_type', 'clean')
        if noise_type is None:
            noise_type = 'clean'
            
        ratio = results.get('unlearnable_ratio', None)
        
        if ratio is not None:
            label = f"{noise_type} ({ratio:.2f} ratio) Train Acc"
        else:
            label = f"{noise_type} Train Acc"
            
        plt.plot(results['epochs'], results['train_acc'], marker='o', linestyle='-', label=label)
    
    plt.title('Training Accuracy (with noise) over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    for results in training_results_list:
        if 'clean_train_acc' in results:
            noise_type = results.get('noise_type', 'clean')
            if noise_type is None:
                noise_type = 'clean'
                
            ratio = results.get('unlearnable_ratio', None)
            
            if ratio is not None:
                label = f"{noise_type} ({ratio:.2f} ratio) Clean Train Acc"
            else:
                label = f"{noise_type} Clean Train Acc"
                
            plt.plot(results['epochs'], results['clean_train_acc'], marker='d', linestyle='-', label=label)
    
    plt.title('Clean Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    for results in training_results_list:
        noise_type = results.get('noise_type', 'clean')
        if noise_type is None:
            noise_type = 'clean'
            
        ratio = results.get('unlearnable_ratio', None)
        
        if ratio is not None:
            label = f"{noise_type} ({ratio:.2f} ratio) Test Acc"
        else:
            label = f"{noise_type} Test Acc"
        
        test_key = 'test_acc' if 'test_acc' in results else 'clean_test_acc'
        plt.plot(results['epochs'], results[test_key], marker='s', linestyle='-', label=label)

    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    
    noise_types = "_".join([str(results.get('noise_type', 'clean')).replace(" ", "_") 
                            for results in training_results_list])
    save_path = f'results/training_curves_{noise_types}.png'
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Training curves saved to {save_path}")
    
def plot_ratio_impact(ratio_results, noise_type):
    plt.figure(figsize=(10, 8))
    
    if noise_type is None:
        noise_type = 'unknown'
    
    ratios = [result['unlearnable_ratio'] for result in ratio_results]
    final_train_accs = [result['train_acc'][-1] for result in ratio_results]
    
    plt.plot(ratios, final_train_accs, 'o-', color='blue', label='Final Training Accuracy (w/ noise)')
    
    if 'clean_train_acc' in ratio_results[0]:
        final_clean_train_accs = [result['clean_train_acc'][-1] for result in ratio_results]
        plt.plot(ratios, final_clean_train_accs, 'd-', color='red', label='Final Clean Train Accuracy')
    
    test_key = 'test_acc' if 'test_acc' in ratio_results[0] else 'clean_test_acc'
    final_test_accs = [result[test_key][-1] for result in ratio_results]
    plt.plot(ratios, final_test_accs, 's-', color='green', label='Final Test Accuracy')
    
    plt.title(f'Impact of {noise_type} Unlearnable Data Ratio on Model Performance')
    plt.xlabel('Unlearnable Data Ratio')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    
    safe_noise_type = str(noise_type).replace(" ", "_")
    save_path = f'results/ratio_impact_{safe_noise_type}.png'
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Ratio impact plot saved to {save_path}")