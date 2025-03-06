import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=(10, 3, 32, 32)):
        torch.manual_seed(self.seed)
        return torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)

    def min_min_attack(self, images, labels, model, criterion):
        images = images.detach()
        delta = torch.zeros_like(images, requires_grad=True).to(device)
        
        for _ in range(self.num_steps):
            perturbed_images = torch.clamp(images + delta, 0, 1)
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)            
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            delta.data = (delta.data - self.step_size * grad.sign()).clamp(-self.epsilon, self.epsilon)

        return delta.detach()

    def min_max_attack(self, images, labels, model, criterion):
        images = images.detach()
        delta = torch.zeros_like(images, requires_grad=True).to(device)

        for _ in range(self.num_steps):
            perturbed_images = torch.clamp(images + delta, 0, 1)
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            delta.data = (delta.data + self.step_size * grad.sign()).clamp(-self.epsilon, self.epsilon)

        return delta.detach()

    def batch_min_min_attack(self, images, labels, noise_batch, model, criterion):
        images = images.to(device)
        labels = labels.to(device)
        noise_batch = noise_batch.clone().detach().to(device).requires_grad_(True)
        batch_size = images.size(0)
        
        for _ in range(self.num_steps):
            perturbed_images = torch.clamp(images + noise_batch, 0, 1)
            outputs = model(perturbed_images)
            individual_losses = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                individual_losses[i] = criterion(outputs[i:i+1], labels[i:i+1])

            total_loss = individual_losses.sum()
            gradients = torch.autograd.grad(total_loss, noise_batch, retain_graph=False)[0]
            
            noise_batch = noise_batch.detach() - self.step_size * gradients.sign()
            noise_batch = torch.clamp(noise_batch, -self.epsilon, self.epsilon)
            noise_batch.requires_grad_(True)
            
        return noise_batch.detach()

    def batch_min_max_attack(self, images, labels, noise_batch, model, criterion):
        images = images.to(device)
        labels = labels.to(device)
        noise_batch = noise_batch.clone().detach().to(device).requires_grad_(True)
        batch_size = images.size(0)
        
        for _ in range(self.num_steps):
            perturbed_images = torch.clamp(images + noise_batch, 0, 1)
            outputs = model(perturbed_images)

            individual_losses = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                individual_losses[i] = criterion(outputs[i:i+1], labels[i:i+1])
            
            total_loss = individual_losses.sum()
            gradients = torch.autograd.grad(total_loss, noise_batch, retain_graph=False)[0]
            noise_batch = noise_batch.detach() + self.step_size * gradients.sign()
            noise_batch = torch.clamp(noise_batch, -self.epsilon, self.epsilon)
            noise_batch.requires_grad_(True)
            
        return noise_batch.detach()