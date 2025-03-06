import torch
import os
from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_unlearnable_dataset(train_loader, noise, noise_type, epsilon=8/255):
    all_images = []
    all_labels = []
    
    print(f"Creating unlearnable dataset with {noise_type} noise...")
    
    if "samplewise" in noise_type and isinstance(noise, dict) and 'images' in noise and 'labels' in noise:
        print("Using preselected samples and noises for samplewise noise")
        images = noise['images'].to(device)
        labels = noise['labels'].to(device)
        noises = noise['noises'].to(device)
        
        noisy_images = images + noises
        noisy_images = torch.clamp(noisy_images, 0, 1)
        
        all_images = [noisy_images.cpu()]
        all_labels = [labels.cpu()]
    else:
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Applying noise")):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            if "classwise" in noise_type and isinstance(noise, torch.Tensor):
                for i, label in enumerate(labels):
                    images[i] += noise[label]
            elif "samplewise" in noise_type and isinstance(noise, dict) and 'noises' in noise:
                batch_start_idx = batch_idx * train_loader.batch_size
                for i in range(len(labels)):
                    idx = batch_start_idx + i
                    if idx < len(noise['noises']):
                        images[i] += noise['noises'][idx].to(device)
            elif noise_type == "random" and isinstance(noise, torch.Tensor):
                if len(noise.shape) == 4 and noise.shape[0] == 10:
                    for i, label in enumerate(labels):
                        images[i] += noise[label % 10]
                else:
                    images += noise                    
            images = torch.clamp(images, 0, 1)
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
    
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    
    unlearnable_dataset = {
        'images': all_images,
        'labels': all_labels,
        'noise_type': noise_type
    }
    
    os.makedirs('data', exist_ok=True)
    save_path = f'data/unlearnable_dataset_{noise_type}.pt'
    torch.save(unlearnable_dataset, save_path)
    print(f"Saved unlearnable dataset with {noise_type} noise to {save_path}")
    print(f"Dataset size: {len(all_images)} samples")
    
    return unlearnable_dataset



def create_mixed_dataset(clean_loader, unlearnable_dataset, ratio=0.5, target_class=None):

    noise_type = unlearnable_dataset['noise_type']
    class_suffix = f"_class{target_class}" if target_class is not None else ""
    saved_path = f'data/mixed_dataset_{noise_type}{class_suffix}_{ratio:.2f}.pt'
    
    print(f"\n====== Creating Mixed Dataset - Debug Info ======")
    print(f"Mixing ratio: {ratio:.2f}")
    print(f"Noise type: {noise_type}")
    if target_class is not None:
        print(f"Target class: {target_class} (only this class will be poisoned)")
    print(f"Save path: {saved_path}")
    

    all_clean_images = []
    all_clean_labels = []
    for images, labels in tqdm(clean_loader, desc="Collecting clean data"):
        all_clean_images.append(images.cpu())
        all_clean_labels.append(labels.cpu())
    
    clean_images = torch.cat(all_clean_images)
    clean_labels = torch.cat(all_clean_labels)
    

    unlearnable_images = unlearnable_dataset['images'].cpu()
    unlearnable_labels = unlearnable_dataset['labels'].cpu()
    
    total_samples = len(clean_images)
    unlearnable_samples = len(unlearnable_images)
    
    print(f"Clean dataset sample count: {total_samples}")
    print(f"Unlearnable dataset sample count: {unlearnable_samples}")
    
    assert unlearnable_samples == total_samples, "Clean dataset and unlearnable dataset must have the same number of samples"
    

    if ratio == 0.0:
        print("Ratio=0, returning clean dataset without shuffling")
        mixed_dataset = {
            'images': clean_images,
            'labels': clean_labels,
            'unlearnable_ratio': 0.0,
            'noise_type': noise_type,
            'unlearnable_indices': torch.tensor([], dtype=torch.long),
            'total_samples': total_samples,
            'target_class': target_class
        }
        
        os.makedirs('data', exist_ok=True)
        torch.save(mixed_dataset, saved_path)
        print(f"Created clean dataset (ratio=0)")
        return mixed_dataset
    

    if target_class is not None:

        target_indices = (clean_labels == target_class).nonzero(as_tuple=True)[0]
        other_indices = (clean_labels != target_class).nonzero(as_tuple=True)[0]
        
        num_target = len(target_indices)
        num_unlearnable = int(num_target * ratio)
        
        print(f"Target class {target_class} has {num_target} samples")
        print(f"Will poison {num_unlearnable} samples of class {target_class}")
        
        torch.manual_seed(42)
        perm = torch.randperm(num_target)
        unlearnable_indices = target_indices[perm[:num_unlearnable]]
 
        all_indices = torch.cat([unlearnable_indices, 
                                 target_indices[perm[num_unlearnable:]], 
                                 other_indices])
    else:

        num_unlearnable = int(total_samples * ratio)
        print(f"Using unlearnable samples: {num_unlearnable}")
        
        torch.manual_seed(42)
        indices = torch.randperm(total_samples)
        unlearnable_indices = indices[:num_unlearnable]
        all_indices = indices
    

    mixed_images = clean_images.clone()
    
    if len(unlearnable_indices) > 0:
        unlearnable_mask = torch.zeros(total_samples, dtype=torch.bool)
        unlearnable_mask[unlearnable_indices] = True
        

        for i in unlearnable_indices:
            mixed_images[i] = unlearnable_images[i]
        
        sample_idx = unlearnable_indices[0].item()
        diff = (mixed_images[sample_idx] - unlearnable_images[sample_idx]).abs().sum().item()
        replaced_success = diff < 0.001
        clean_diff = (mixed_images[sample_idx] - clean_images[sample_idx]).abs().sum().item()
        print(f"Sample {sample_idx} - Difference from unlearnable: {diff:.6f}, difference from clean: {clean_diff:.6f}")
        print(f"Replacement validation: {'Success' if replaced_success else 'Failed'}")
    

    mixed_dataset = {
        'images': mixed_images,
        'labels': clean_labels,
        'unlearnable_ratio': ratio,
        'noise_type': noise_type,
        'unlearnable_indices': unlearnable_indices,
        'total_samples': total_samples,
        'target_class': target_class
    }
    os.makedirs('data', exist_ok=True)
    torch.save(mixed_dataset, saved_path)
    
    if target_class is not None:
        print(f"Created mixed dataset with {ratio:.2f} ratio of {noise_type} noise for class {target_class}")
        print(f"Total samples: {total_samples}, of which {len(unlearnable_indices)} are unlearnable (all in class {target_class})")
    else:
        print(f"Created mixed dataset with {ratio:.2f} ratio of {noise_type} noise")
        print(f"Total samples: {total_samples}, of which {len(unlearnable_indices)} are unlearnable")
    
    print("====== Mixed Dataset Debug Info End ======\n")
    
    return mixed_dataset


# def create_mixed_dataset(clean_loader, unlearnable_dataset, ratio=0.5):
#     saved_path = f'data/mixed_dataset_{unlearnable_dataset["noise_type"]}_{ratio:.2f}.pt'
    
#     print(f"\n====== Creating Mixed Dataset - Debug Info ======")
#     print(f"Mixing ratio: {ratio:.2f}")
#     print(f"Noise type: {unlearnable_dataset['noise_type']}")
#     print(f"Save path: {saved_path}")
    
#     unlearnable_images = unlearnable_dataset['images'].cpu()
#     unlearnable_labels = unlearnable_dataset['labels'].cpu()
    
#     all_clean_images = []
#     all_clean_labels = []
#     for images, labels in tqdm(clean_loader, desc="Collecting clean data"):
#         all_clean_images.append(images.cpu())
#         all_clean_labels.append(labels.cpu())
    
#     clean_images = torch.cat(all_clean_images)
#     clean_labels = torch.cat(all_clean_labels)
    
#     total_samples = len(clean_images)
#     unlearnable_samples = len(unlearnable_images)
    
#     print(f"Clean dataset sample count: {total_samples}")
#     print(f"Unlearnable dataset sample count: {unlearnable_samples}")
    
#     assert unlearnable_samples == total_samples, "Clean dataset and unlearnable dataset must have the same number of samples"
    
#     num_unlearnable = int(total_samples * ratio)
#     print(f"Using unlearnable samples: {num_unlearnable}")
    
#     torch.manual_seed(42)
#     indices = torch.randperm(total_samples)
#     unlearnable_indices = indices[:num_unlearnable]
#     clean_indices = indices[num_unlearnable:]
    
#     print(f"Unlearnable sample indices count: {len(unlearnable_indices)}")
#     print(f"Clean sample indices count: {len(clean_indices)}")
    
#     mixed_images = clean_images.clone()
    
#     if num_unlearnable > 0:
#         mixed_images[unlearnable_indices] = unlearnable_images[unlearnable_indices]
        
#         replaced_success = False
#         if len(unlearnable_indices) > 0:
#             sample_idx = unlearnable_indices[0].item()
#             diff = (mixed_images[sample_idx] - unlearnable_images[sample_idx]).abs().sum().item()
#             replaced_success = diff < 0.001
#             clean_diff = (mixed_images[sample_idx] - clean_images[sample_idx]).abs().sum().item()
#             print(f"Sample {sample_idx} - Difference from unlearnable: {diff:.6f}, difference from clean: {clean_diff:.6f}")
#             print(f"Replacement validation: {'Success' if replaced_success else 'Failed'}")
    
#     mixed_dataset = {
#         'images': mixed_images,
#         'labels': clean_labels,
#         'unlearnable_ratio': ratio,
#         'noise_type': unlearnable_dataset['noise_type'],
#         'unlearnable_indices': unlearnable_indices,
#         'total_samples': total_samples
#     }
#     if ratio == 0.0 and clean_train_loader is not None:
#         print("Ratio=0: Using original clean_train_loader directly")
#         return clean_train_loader 

# #     if ratio == 0.0:
# #         is_equal = torch.all(mixed_images == clean_images).item()
# #         print(f"Data completely identical at ratio 0.0: {is_equal}")
        
# #         if not is_equal:
# #             diff_count = (mixed_images != clean_images).sum().item()
# #             diff_percentage = diff_count / (mixed_images.numel()) * 100
# #             print(f"Different elements count: {diff_count}, difference percentage: {diff_percentage:.6f}%")
    
# #     os.makedirs('data', exist_ok=True)
# #     torch.save(mixed_dataset, saved_path)
# #     print(f"Created mixed dataset with {ratio:.2f} ratio of {unlearnable_dataset['noise_type']} noise")
# #     print(f"Total samples: {total_samples}, of which {num_unlearnable} are unlearnable")
    
# #     print("====== Mixed Dataset Debug Info End ======\n")
    
#     return mixed_dataset

def create_limited_training_loader(train_loader, noise, noise_type, samples_per_class=None):
    print(f"Creating limited training loader with {noise_type} noise...")
    
    if "samplewise" in noise_type and isinstance(noise, dict) and 'images' in noise and 'labels' in noise and 'noises' in noise:
        print("Using precomputed samplewise noise")
        
        images = noise['images']
        labels = noise['labels']
        noises = noise['noises']
        
        noisy_images = images + noises
        noisy_images = torch.clamp(noisy_images, 0, 1)
        
        dataset = TensorDataset(noisy_images, labels)
        import os
        num_workers = min(4, os.cpu_count() or 1)
        loader = DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
        
        print(f"Created limited training loader with {len(noisy_images)} samples")
        return loader
    else:
        class_images = {i: [] for i in range(10)}
        class_labels = {i: [] for i in range(10)}
        class_counts = {i: 0 for i in range(10)}
        
        max_samples = samples_per_class if samples_per_class else float('inf')
        
        for images, labels in tqdm(train_loader, desc="Selecting samples by class"):
            for i, label in enumerate(labels):
                label_idx = label.item()
                if class_counts[label_idx] < max_samples:
                    img = images[i:i+1]
                    
                    if "classwise" in noise_type and isinstance(noise, torch.Tensor):
                        noisy_img = img + noise[label_idx].cpu()
                    elif noise_type == "random" and isinstance(noise, torch.Tensor):
                        if len(noise.shape) == 4 and noise.shape[0] == 10:
                            noisy_img = img + noise[label_idx].cpu()
                        else:
                            noisy_img = img + noise.cpu()
                    else:
                        noisy_img = img
                    
                    noisy_img = torch.clamp(noisy_img, 0, 1)
                    
                    class_images[label_idx].append(noisy_img)
                    class_labels[label_idx].append(torch.tensor([label_idx]))
                    class_counts[label_idx] += 1
            
            if all(count >= max_samples for count in class_counts.values()):
                break
        
        all_images = []
        all_labels = []
        for label_idx in range(10):
            if class_images[label_idx]:
                all_images.append(torch.cat(class_images[label_idx]))
                all_labels.append(torch.cat(class_labels[label_idx]))
        
        all_images = torch.cat(all_images) if all_images else torch.tensor([])
        all_labels = torch.cat(all_labels) if all_labels else torch.tensor([])
        
        if len(all_images) > 0:
            perm = torch.randperm(len(all_images))
            all_images = all_images[perm]
            all_labels = all_labels[perm]
        
        dataset = TensorDataset(all_images, all_labels)
        import os
        num_workers = min(4, os.cpu_count() or 1)
        loader = DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
        
        print(f"Created limited training loader with {len(all_images)} samples")
        return loader