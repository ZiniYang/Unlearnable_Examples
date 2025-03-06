import torch
import torch.nn as nn
import os
import numpy as np
import time
from tqdm import tqdm
from toolbox import PerturbationTool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_classwise_errmin_noise(train_loader, model, epsilon=8/255, steps=20, alpha=2/255):
    print(f"Parameters: epsilon={epsilon}, steps={steps}, alpha={alpha}")

    model.eval()
    ptool = PerturbationTool(seed=0, epsilon=epsilon, num_steps=steps, step_size=alpha)
    criterion = nn.CrossEntropyLoss()

    noise = torch.zeros(10, 3, 32, 32).to(device)
    class_counts = torch.zeros(10).to(device)
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
        images, labels = images.to(device), labels.to(device)
        
        delta = ptool.min_min_attack(images, labels, model, criterion)
        for c in range(10):
            mask = (labels == c)
            if mask.sum() > 0:
                noise[c] += delta[mask].sum(dim=0)
                class_counts[c] += mask.sum()
    for c in range(10):
        if class_counts[c] > 0:
            noise[c] /= class_counts[c]
    noise = torch.clamp(noise, -epsilon, epsilon)
    os.makedirs('data', exist_ok=True)
    save_path = 'data/classwise_errmin_noise.pt'
    torch.save(noise, save_path)
    
    end_time = time.time()
    print(f"\nSaved Classwise Error-Minimizing noise to {save_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return noise.detach()

def generate_classwise_errmax_noise(train_loader, model, epsilon=8/255, steps=20, alpha=2/255):
    print("\nGenerating classwise error-maximizing noise (PGD)...")
    print(f"Parameters: epsilon={epsilon}, steps={steps}, alpha={alpha}")

    model.eval()
    ptool = PerturbationTool(seed=0, epsilon=epsilon, num_steps=steps, step_size=alpha)
    criterion = nn.CrossEntropyLoss()

    noise = torch.zeros(10, 3, 32, 32).to(device)
    class_counts = torch.zeros(10).to(device)

    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
        images, labels = images.to(device), labels.to(device)
        
        delta = ptool.min_max_attack(images, labels, model, criterion)
        for c in range(10):
            mask = (labels == c)
            if mask.sum() > 0:
                noise[c] += delta[mask].sum(dim=0)
                class_counts[c] += mask.sum()

    for c in range(10):
        if class_counts[c] > 0:
            noise[c] /= class_counts[c]

    noise = torch.clamp(noise, -epsilon, epsilon)
    os.makedirs('data', exist_ok=True)
    save_path = 'data/classwise_errmax_noise.pt'
    torch.save(noise, save_path)
    
    end_time = time.time()
    print(f"\nSaved Classwise Error-Maximizing noise to {save_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return noise.detach()

def generate_samplewise_errmin_noise(train_loader, model, epsilon=8/255, steps=20, alpha=2/255, samples_per_class=None):
    print(f"\nGenerating samplewise error-minimizing noise (batch processing)...")
    if samples_per_class:
        print(f"Limiting to {samples_per_class} samples per class")
    print(f"Parameters: epsilon={epsilon}, steps={steps}, alpha={alpha}")

    model.eval()
    ptool = PerturbationTool(seed=0, epsilon=epsilon, num_steps=steps, step_size=alpha)
    criterion = nn.CrossEntropyLoss(reduction='none')

    all_noises = []
    all_images = []
    all_labels = []

    np.random.seed(0)

    if samples_per_class:
        class_samples = {i: 0 for i in range(10)}
        total_selected = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Counting samples")):
            for label in labels:
                label_idx = label.item()
                if class_samples[label_idx] < samples_per_class:
                    class_samples[label_idx] += 1
                    total_selected += 1
            
            if all(count >= samples_per_class for count in class_samples.values()):
                break
        
        class_samples = {i: 0 for i in range(10)}
        
        selected_noises = torch.zeros((total_selected, 3, 32, 32))
        selected_images = torch.zeros((total_selected, 3, 32, 32))
        selected_labels = torch.zeros(total_selected, dtype=torch.long)
        
        current_idx = 0
        print("Processing batches for selected samples...")
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
            batch_selected_indices = []
            batch_selected_labels = []
            
            for i, label in enumerate(labels):
                label_idx = label.item()
                if class_samples[label_idx] < samples_per_class:
                    batch_selected_indices.append(i)
                    batch_selected_labels.append(label_idx)
                    class_samples[label_idx] += 1
            
            if not batch_selected_indices:
                continue 
            batch_images = images[batch_selected_indices]
            batch_labels = labels[batch_selected_indices]
            
            batch_noise = torch.FloatTensor(len(batch_selected_indices), 3, 32, 32).uniform_(-epsilon, epsilon)
            optimized_noise = ptool.batch_min_min_attack(batch_images, batch_labels, batch_noise, model, nn.CrossEntropyLoss())
            num_selected = len(batch_selected_indices)
            selected_images[current_idx:current_idx+num_selected] = batch_images.cpu()
            selected_labels[current_idx:current_idx+num_selected] = batch_labels.cpu()
            selected_noises[current_idx:current_idx+num_selected] = optimized_noise.cpu()
            
            current_idx += num_selected
            if all(count >= samples_per_class for count in class_samples.values()):
                break

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        all_images = [selected_images[:current_idx]]
        all_labels = [selected_labels[:current_idx]]
        all_noises = [selected_noises[:current_idx]]
        
    else:
        print("Processing all samples by batch...")
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
            batch_size = images.size(0)
            batch_noise = torch.FloatTensor(batch_size, 3, 32, 32).uniform_(-epsilon, epsilon)
            optimized_noise = ptool.batch_min_min_attack(images, labels, batch_noise, model, nn.CrossEntropyLoss())
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_noises.append(optimized_noise.cpu())

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    final_images = torch.cat(all_images)
    final_labels = torch.cat(all_labels)
    final_noises = torch.cat(all_noises)

    noise_dict = {
        'noises': final_noises,
        'images': final_images,
        'labels': final_labels
    }
    
    os.makedirs('data', exist_ok=True)
    if samples_per_class:
        save_path = f'data/samplewise_errmin_noise_{samples_per_class}_per_class.pt'
    else:
        save_path = 'data/samplewise_errmin_noise_all.pt'
    torch.save(noise_dict, save_path)
    
    end_time = time.time()
    print(f"\nSaved Samplewise Error-Minimizing noise to {save_path}")
    print(f"Total samples with noise: {len(final_noises)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    return noise_dict

def generate_samplewise_errmax_noise(train_loader, model, epsilon=8/255, steps=20, alpha=2/255, samples_per_class=None):
    print(f"\nGenerating samplewise error-maximizing noise (batch processing)...")
    if samples_per_class:
        print(f"Limiting to {samples_per_class} samples per class")
    print(f"Parameters: epsilon={epsilon}, steps={steps}, alpha={alpha}")

    model.eval()
    ptool = PerturbationTool(seed=0, epsilon=epsilon, num_steps=steps, step_size=alpha)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    all_noises = []
    all_images = []
    all_labels = []
    
    np.random.seed(0)
    
    if samples_per_class:
        class_samples = {i: 0 for i in range(10)}
        total_selected = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Counting samples")):
            for label in labels:
                label_idx = label.item()
                if class_samples[label_idx] < samples_per_class:
                    class_samples[label_idx] += 1
                    total_selected += 1
            
            if all(count >= samples_per_class for count in class_samples.values()):
                break
        
        class_samples = {i: 0 for i in range(10)}
        
        selected_noises = torch.zeros((total_selected, 3, 32, 32))
        selected_images = torch.zeros((total_selected, 3, 32, 32))
        selected_labels = torch.zeros(total_selected, dtype=torch.long)

        current_idx = 0
        print("Processing batches for selected samples...")
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
            batch_selected_indices = []
            batch_selected_labels = []
            
            for i, label in enumerate(labels):
                label_idx = label.item()
                if class_samples[label_idx] < samples_per_class:
                    batch_selected_indices.append(i)
                    batch_selected_labels.append(label_idx)
                    class_samples[label_idx] += 1
            
            if not batch_selected_indices:
                continue  
            batch_images = images[batch_selected_indices]
            batch_labels = labels[batch_selected_indices]
            
            batch_noise = torch.FloatTensor(len(batch_selected_indices), 3, 32, 32).uniform_(-epsilon, epsilon)
            optimized_noise = ptool.batch_min_max_attack(batch_images, batch_labels, batch_noise, model, nn.CrossEntropyLoss())
            num_selected = len(batch_selected_indices)
            selected_images[current_idx:current_idx+num_selected] = batch_images.cpu()
            selected_labels[current_idx:current_idx+num_selected] = batch_labels.cpu()
            selected_noises[current_idx:current_idx+num_selected] = optimized_noise.cpu()
            
            current_idx += num_selected

            if all(count >= samples_per_class for count in class_samples.values()):
                break

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        all_images = [selected_images[:current_idx]]
        all_labels = [selected_labels[:current_idx]]
        all_noises = [selected_noises[:current_idx]]
        
    else:
        print("Processing all samples by batch...")
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
            batch_size = images.size(0)
            batch_noise = torch.FloatTensor(batch_size, 3, 32, 32).uniform_(-epsilon, epsilon)
            optimized_noise = ptool.batch_min_max_attack(images, labels, batch_noise, model, nn.CrossEntropyLoss())
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_noises.append(optimized_noise.cpu())
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    final_images = torch.cat(all_images)
    final_labels = torch.cat(all_labels)
    final_noises = torch.cat(all_noises)

    noise_dict = {
        'noises': final_noises,
        'images': final_images,
        'labels': final_labels
    }
    
    os.makedirs('data', exist_ok=True)
    if samples_per_class:
        save_path = f'data/samplewise_errmax_noise_{samples_per_class}_per_class.pt'
    else:
        save_path = 'data/samplewise_errmax_noise_all.pt'
    torch.save(noise_dict, save_path)
    
    end_time = time.time()
    print(f"\nSaved Samplewise Error-Maximizing noise to {save_path}")
    print(f"Total samples with noise: {len(final_noises)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    return noise_dict

def generate_random_noise(noise_shape=(10, 3, 32, 32), epsilon=8/255):
    print("\nGenerating random noise...")
    print(f"Parameters: epsilon={epsilon}, shape={noise_shape}")
    
    ptool = PerturbationTool(seed=0, epsilon=epsilon, num_steps=1, step_size=0.0)
    noise = ptool.random_noise(noise_shape=noise_shape)
    
    os.makedirs('data', exist_ok=True)
    save_path = 'data/random_noise.pt'
    torch.save(noise, save_path)
    print(f"Saved Random noise to {save_path}")
    
    return noise