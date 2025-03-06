import torch
from torch.utils.data import DataLoader, TensorDataset
from data.cifar10 import get_cifar10
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dataset(loader, model, noise=None, noise_type="None", apply_noise=False):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Evaluating dataset", leave=False)):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            
            if apply_noise and noise is not None:
                if "classwise" in noise_type and isinstance(noise, torch.Tensor):
                    for i, label in enumerate(labels):
                        images[i] += noise[label]
                elif "samplewise" in noise_type:
                    if isinstance(noise, dict) and 'noises' in noise:
                        batch_start_idx = batch_idx * batch_size
                        for i in range(batch_size):
                            idx = batch_start_idx + i
                            if idx < len(noise['noises']):
                                images[i] += noise['noises'][idx].to(device)
                    elif isinstance(noise, torch.Tensor) and batch_idx < len(noise):
                        batch_noise = noise[batch_idx].to(device)
                        valid_samples = min(batch_size, batch_noise.shape[0])
                        images[:valid_samples] += batch_noise[:valid_samples]
                elif noise_type == "random" and isinstance(noise, torch.Tensor):
                    if len(noise.shape) == 4 and noise.shape[0] == 10:
                        for i, label in enumerate(labels):
                            images[i] += noise[label % 10]
                    else:
                        images += noise
                
                images = torch.clamp(images, 0, 1)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def evaluate_model_on_dataset(model, dataset_path=None):
    model.eval()
    
    if dataset_path:
        loaded_data = torch.load(dataset_path)
        images = loaded_data['images'].to(device)
        labels = loaded_data['labels'].to(device)
        test_dataset = TensorDataset(images, labels)
        
        import os
        num_workers = min(4, os.cpu_count() or 1)
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
    else:
        _, test_loader = get_cifar10(batch_size=128, num_workers=4)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on clean test set", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy