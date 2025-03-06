# adversarial_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from toolbox import PerturbationTool
from data.cifar10 import get_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdversarialTrainer:
    def __init__(self, model, train_loader, test_loader, 
                 epsilon=8/255, pgd_steps=10, pgd_alpha=2/255):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def _generate_adversarial_examples(self, images, labels):
        """
        Generate adversarial examples using PGD attack
        """
        # Convert images to float and normalize if needed
        images = images.clone().detach().to(device).float()
        
        # Initialize perturbation within allowed epsilon
        delta = torch.zeros_like(images, requires_grad=True)
        
        for step in range(self.pgd_steps):
            # Forward pass
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, 0, 1)
            
            # Clear previous gradients
            if delta.grad is not None:
                delta.grad.zero_()
                
            # Compute loss
            outputs = self.model(adv_images)
            loss = -self.criterion(outputs, labels)  # Negative because we want to maximize loss
            
            # Compute gradients
            loss.backward()
            
            # Update perturbation with gradient ascent
            grad = delta.grad.detach()
            delta.data = delta.data - self.pgd_alpha * grad.sign()  # Gradient ascent
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            
        return torch.clamp(images + delta.detach(), 0, 1)
    
    def evaluate(self, attack_type=None):
        """
        Evaluate model on clean and adversarial examples
        """
        self.model.eval()
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for images, labels in self.test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total += batch_size
            
            # Clean accuracy
            with torch.no_grad():
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                clean_correct += (predicted == labels).sum().item()
            
            # Adversarial accuracy
            if attack_type:
                self.model.eval()
                adv_images = self._generate_adversarial_examples(images, labels)
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    adv_correct += (predicted == labels).sum().item()
                
        clean_acc = 100 * clean_correct / total
        adv_acc = 100 * adv_correct / total if attack_type else None
        
        return clean_acc, adv_acc

    def train_epoch(self, epoch, attack_type):
        """
        Train for one epoch
        """
        self.model.train()
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total += batch_size
            
            # Generate adversarial examples
            self.optimizer.zero_grad()
            adv_images = self._generate_adversarial_examples(images, labels)
            
            # Forward pass
            outputs_clean = self.model(images)
            outputs_adv = self.model(adv_images)
            
            # Compute loss
            loss = (
                0.5 * self.criterion(outputs_clean, labels) +
                0.5 * self.criterion(outputs_adv, labels)
            )
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs_adv.data, 1)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        return train_acc
    
    def train(self, epochs, attack_type, save_path=None):
        """
        Conduct complete adversarial training
        """
        print(f"\nStarting adversarial training against {attack_type} attack:")
        print(f"Parameters: epsilon={self.epsilon}, steps={self.pgd_steps}, alpha={self.pgd_alpha}")
        
        results = {
            'epochs': [],
            'train_acc': [],
            'adv_test_acc': [],
            'clean_test_acc': []
        }
        
        for epoch in range(epochs):
            # Train for one epoch
            train_acc = self.train_epoch(epoch, attack_type)
            
            # Evaluate every 5 epochs or at the end
            if epoch % 5 == 0 or epoch == epochs - 1:
                clean_test_acc, adv_test_acc = self.evaluate(attack_type)
                
                results['epochs'].append(epoch + 1)
                results['train_acc'].append(train_acc)
                results['adv_test_acc'].append(adv_test_acc)
                results['clean_test_acc'].append(clean_test_acc)
                
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Adv Test Acc: {adv_test_acc:.2f}% | "
                      f"Clean Test Acc: {clean_test_acc:.2f}%")
            
            self.scheduler.step()
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        return results