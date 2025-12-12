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


def compute_label_flip_asr(
    model: nn.Module,
    data_loader: DataLoader,
    source_class: int,
    target_class: int,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute ASR for label flip attacks.
    
    For label flip attacks, ASR measures:
    1. What % of source_class samples are misclassified as target_class
    2. How much the source_class accuracy dropped
    
    Args:
        model: The model to evaluate
        data_loader: Clean test data loader
        source_class: The class being flipped from
        target_class: The class being flipped to
        device: Device for computation
        
    Returns:
        Dictionary with ASR metrics
    """
    model.eval()
    model.to(device)
    
    source_total = 0
    source_correct = 0
    source_to_target = 0  # Misclassified as target
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Focus on source class samples
            source_mask = labels == source_class
            source_preds = predicted[source_mask]
            source_labels = labels[source_mask]
            
            source_total += source_mask.sum().item()
            source_correct += (source_preds == source_labels).sum().item()
            source_to_target += (source_preds == target_class).sum().item()
    
    source_accuracy = source_correct / source_total if source_total > 0 else 0.0
    flip_rate = source_to_target / source_total if source_total > 0 else 0.0
    
    return {
        "source_accuracy": source_accuracy,
        "flip_rate": flip_rate,  # This is the ASR for label flip
        "source_total": source_total,
        "source_correct": source_correct,
        "misclassified_as_target": source_to_target
    }


def compute_model_poisoning_metrics(
    poisoned_model: nn.Module,
    clean_model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute metrics for model poisoning attacks.
    
    Measures:
    1. Accuracy drop compared to clean model
    2. Parameter divergence from clean model
    3. Prediction disagreement rate
    
    Args:
        poisoned_model: Model trained with poisoning attack
        clean_model: Model trained without attack (baseline)
        data_loader: Test data loader
        device: Device for computation
        
    Returns:
        Dictionary with model poisoning metrics
    """
    poisoned_model.eval()
    clean_model.eval()
    poisoned_model.to(device)
    clean_model.to(device)
    
    # Compute accuracy for both models
    poisoned_correct = 0
    clean_correct = 0
    disagreements = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            poisoned_out = poisoned_model(images)
            clean_out = clean_model(images)
            
            _, poisoned_pred = torch.max(poisoned_out.data, 1)
            _, clean_pred = torch.max(clean_out.data, 1)
            
            total += labels.size(0)
            poisoned_correct += (poisoned_pred == labels).sum().item()
            clean_correct += (clean_pred == labels).sum().item()
            disagreements += (poisoned_pred != clean_pred).sum().item()
    
    poisoned_acc = poisoned_correct / total if total > 0 else 0.0
    clean_acc = clean_correct / total if total > 0 else 0.0
    disagreement_rate = disagreements / total if total > 0 else 0.0
    
    # Compute parameter divergence
    param_divergence = compute_param_divergence(poisoned_model, clean_model)
    
    return {
        "poisoned_accuracy": poisoned_acc,
        "clean_accuracy": clean_acc,
        "accuracy_drop": clean_acc - poisoned_acc,
        "disagreement_rate": disagreement_rate,
        "param_divergence": param_divergence
    }


def compute_param_divergence(
    model1: nn.Module,
    model2: nn.Module
) -> float:
    """
    Compute L2 distance between model parameters.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        L2 distance between parameters
    """
    total_diff = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        total_diff += torch.norm(p1.data - p2.data).item() ** 2
    return total_diff ** 0.5


class AttackMetricsTracker:
    """
    Unified tracker for attack success metrics.
    
    Provides consistent interface for measuring attack effectiveness
    across different attack types.
    
    Usage:
        tracker = AttackMetricsTracker(attack_type="backdoor", target_class=0)
        metrics = tracker.compute(model, test_loader, triggered_loader, device)
    """
    
    def __init__(
        self,
        attack_type: str,
        source_class: int = None,
        target_class: int = None
    ):
        self.attack_type = attack_type
        self.source_class = source_class
        self.target_class = target_class
        self.metrics_history = []
    
    def compute(
        self,
        model: nn.Module,
        clean_loader: DataLoader,
        poisoned_loader: DataLoader = None,
        device: str = "cpu",
        clean_model: nn.Module = None
    ) -> Dict[str, float]:
        """
        Compute attack-specific metrics.
        
        Args:
            model: Model to evaluate
            clean_loader: Clean test data
            poisoned_loader: Triggered/poisoned test data (for backdoor)
            device: Computation device
            clean_model: Baseline model (for model poisoning comparison)
            
        Returns:
            Dictionary with attack metrics
        """
        # Always compute main task accuracy
        main_metrics = evaluate_model(model, clean_loader, device)
        
        result = {
            "main_accuracy": main_metrics["accuracy"],
            "main_loss": main_metrics["loss"]
        }
        
        if self.attack_type == "backdoor" and poisoned_loader is not None:
            asr = compute_attack_success_rate(
                model, poisoned_loader, self.target_class, device
            )
            result["asr"] = asr
            result["attack_type"] = "backdoor"
            
        elif self.attack_type == "label_flip":
            flip_metrics = compute_label_flip_asr(
                model, clean_loader, self.source_class, self.target_class, device
            )
            result.update(flip_metrics)
            result["asr"] = flip_metrics["flip_rate"]
            result["attack_type"] = "label_flip"
            
        elif self.attack_type in ["model_replacement", "model_poisoning", "scaling"]:
            if clean_model is not None:
                poison_metrics = compute_model_poisoning_metrics(
                    model, clean_model, clean_loader, device
                )
                result.update(poison_metrics)
                result["asr"] = poison_metrics["accuracy_drop"]
                result["attack_type"] = "model_poisoning"
            else:
                result["asr"] = None
                result["attack_type"] = "model_poisoning"
        else:
            result["asr"] = None
            result["attack_type"] = self.attack_type
        
        self.metrics_history.append(result)
        return result
    
    def get_history(self) -> list:
        """Get all computed metrics."""
        return self.metrics_history
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from all rounds."""
        if not self.metrics_history:
            return {}
        
        asrs = [m.get("asr") for m in self.metrics_history if m.get("asr") is not None]
        accs = [m.get("main_accuracy") for m in self.metrics_history]
        
        return {
            "final_asr": asrs[-1] if asrs else None,
            "avg_asr": np.mean(asrs) if asrs else None,
            "final_accuracy": accs[-1] if accs else None,
            "avg_accuracy": np.mean(accs) if accs else None
        }


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
