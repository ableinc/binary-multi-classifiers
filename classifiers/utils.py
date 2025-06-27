import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Any
import numpy as np

def setup_device():
    """Setup device and check for multi-GPU availability."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device('cuda')
        return device, True
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, False

def split_data(texts: List[str], labels: List[int], test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    """
    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the remaining data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, 
                   train_loss: float, val_loss: float, save_dir: str, model_name: str):
    """Save model checkpoint with training state."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(save_dir, f"{model_name}_checkpoint.pt"))

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str) -> Tuple[int, float, float]:
    """Load model checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    from collections import Counter
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    class_weights = []
    for i in range(num_classes):
        if i in label_counts:
            weight = total_samples / (num_classes * label_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def print_training_info(epoch: int, total_epochs: int, train_loss: float, val_loss: float, 
                       train_acc: float = 0.0, val_acc: float = 0.0):
    """Print formatted training information."""
    info = f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    if train_acc is not None and val_acc is not None:
        info += f", Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
    print(info)

__all__ = ["setup_device"]
