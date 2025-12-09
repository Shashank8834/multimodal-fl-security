"""
Evaluation Metrics

Functions for evaluating model performance and attack effectiveness.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model accuracy and loss.
    
    Args:
        model: The model to evaluate
        data_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with accuracy and loss metrics
    """
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total
    }


def compute_attack_success_rate(
    model: nn.Module,
    poisoned_loader: DataLoader,
    target_class: int,
    device: str = "cpu"
) -> float:
    """
    Compute attack success rate for backdoor/targeted attacks.
    
    Attack success rate = % of poisoned samples classified as target class
    
    Args:
        model: The model to evaluate
        poisoned_loader: Data loader with poisoned samples
        target_class: The target class for the attack
        device: Device for computation
        
    Returns:
        Attack success rate (0.0 to 1.0)
    """
    model.eval()
    model.to(device)
    
    total = 0
    success = 0
    
    with torch.no_grad():
        for images, _ in poisoned_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += images.size(0)
            success += (predicted == target_class).sum().item()
    
    return success / total if total > 0 else 0.0


def compute_class_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int = 10,
    device: str = "cpu"
) -> Dict[int, float]:
    """
    Compute per-class accuracy.
    
    Args:
        model: The model to evaluate
        data_loader: Test data loader
        num_classes: Number of classes
        device: Device for computation
        
    Returns:
        Dictionary mapping class index to accuracy
    """
    model.eval()
    model.to(device)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    return {
        i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    }


def compute_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int = 10,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        model: The model to evaluate
        data_loader: Test data loader
        num_classes: Number of classes
        device: Device for computation
        
    Returns:
        Confusion matrix as numpy array (true x predicted)
    """
    model.eval()
    model.to(device)
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion[t.long(), p.long()] += 1
    
    return confusion


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of model gradients.
    
    Useful for detecting anomalous updates in FL.
    
    Args:
        model: Model with computed gradients
        
    Returns:
        L2 norm of all gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_update_similarity(
    update1: list,
    update2: list
) -> float:
    """
    Compute cosine similarity between two model updates.
    
    Args:
        update1: First update as list of numpy arrays
        update2: Second update as list of numpy arrays
        
    Returns:
        Cosine similarity (-1.0 to 1.0)
    """
    # Flatten updates
    flat1 = np.concatenate([u.flatten() for u in update1])
    flat2 = np.concatenate([u.flatten() for u in update2])
    
    # Compute cosine similarity
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    from models.simple_cnn import SimpleCNN
    from data_loader import load_mnist, create_data_loaders, get_client_data
    
    # Load data and model
    train_data, test_data = load_mnist("./data")
    train_loader, test_loader = create_data_loaders(
        get_client_data(train_data, 0, 5, "iid"),
        test_data,
        batch_size=64
    )
    
    # Create untrained model
    model = SimpleCNN()
    
    print("Evaluating untrained model...")
    metrics = evaluate_model(model, test_loader)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    print("\nPer-class accuracy:")
    class_acc = compute_class_accuracy(model, test_loader)
    for cls, acc in class_acc.items():
        print(f"  Class {cls}: {acc:.2%}")
