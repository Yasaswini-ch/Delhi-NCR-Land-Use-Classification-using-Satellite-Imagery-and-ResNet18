"""
Training module for CNN-based air quality prediction.
Handles training loop, validation, model saving, and logging.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, load_config
from src.q3_model import AQICNN, create_model, count_parameters, get_model_summary


class AQIDataset(Dataset):
    """
    Dataset class for AQI prediction.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            features: Feature array
            labels: Label array
            transform: Optional transform
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def prepare_data_for_cnn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare data for CNN training.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, test_loader, num_features)
    """
    logger = setup_logger("Train")
    
    exclude_cols = ['station_id', 'station_name', 'date', 'aqi', 'aqi_category', 
                    'pm25', 'pm10', 'no2', 'o3', 'label', 'longitude_x', 'latitude_x', 
                    'longitude_y', 'latitude_y', 'image_name']
    
    available_cols = [c for c in train_df.columns 
                      if c not in exclude_cols 
                      and train_df[c].dtype in ['int64', 'float64', 'float32']]
    
    if not available_cols:
        logger.warning("No feature columns found")
        available_cols = [c for c in train_df.columns 
                         if train_df[c].dtype in ['int64', 'float64', 'float32']]
    
    logger.info(f"Using features: {available_cols}")
    num_features = len(available_cols)
    
    X_train = train_df[available_cols].values
    X_test = test_df[available_cols].values
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    batch_size = config['training']['batch_size']
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, num_features


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: CNN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device
        logger: Logger
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: CNN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        logger: Logger
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Complete training loop with early stopping.
    
    Args:
        model: CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Computing device
        logger: Logger
    
    Returns:
        Tuple of (trained model, history dictionary)
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'step')
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    
    num_epochs = config['training']['num_epochs']
    early_stopping = config['training'].get('early_stopping', {})
    patience = early_stopping.get('patience', 10)
    min_delta = early_stopping.get('min_delta', 0.001)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    
    output_path = Path(config['paths']['outputs'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, logger
        )
        
        if scheduler_type == 'step':
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            if config['training'].get('save_best_model', True):
                best_model_state = model.state_dict().copy()
                save_path = output_path / 'best_model.pth'
                torch.save(best_model_state, save_path)
                logger.info(f"Best model saved: {save_path}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def plot_training_history(history: Dict[str, list], config: dict) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        config: Configuration dictionary
    """
    import matplotlib.pyplot as plt
    
    logger = setup_logger("Train")
    output_path = Path(config['paths']['outputs'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    save_path = output_path / 'training_history.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved: {save_path}")


def run_training(config_path: Optional[str] = None) -> dict:
    """
    Main training function.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Dictionary with training results
    """
    logger = setup_logger("Train")
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / 'configs' / 'config.yaml')
    
    config = load_config(config_path)
    
    seed = config['dataset']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    processed_path = Path(config['paths']['processed_data'])
    train_df = pd.read_csv(processed_path / 'train.csv')
    test_df = pd.read_csv(processed_path / 'test.csv')
    logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")
    
    val_split = config['dataset'].get('validation_split', 0.2)
    
    label_counts = train_df['label'].value_counts()
    min_label_count = label_counts.min()
    
    if min_label_count >= 2:
        train_data, val_data = train_test_split_with_stratify(
            train_df,
            test_size=val_split,
            random_state=seed,
            stratify=train_df['label']
        )
    else:
        logger.warning("Cannot stratify - some classes have less than 2 samples. Using random split.")
        train_data, val_data = train_test_split_with_stratify(
            train_df,
            test_size=val_split,
            random_state=seed,
            stratify=None
        )
    
    logger.info(f"Split train: {len(train_data)}, val: {len(val_data)}")
    
    train_loader, val_loader, num_features = prepare_data_for_cnn(train_data, val_data, config)
    
    logger.info(f"Number of features: {num_features}")
    
    model = create_model(config, input_dim=num_features)
    model = model.to(device)
    
    model_summary = get_model_summary(model)
    logger.info(f"Model parameters: {model_summary['trainable_parameters']:,}")
    
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        device,
        logger
    )
    
    plot_training_history(history, config)
    
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    
    results = {
        'final_val_accuracy': final_val_acc,
        'best_val_accuracy': max(history['val_acc']) if history['val_acc'] else 0,
        'total_epochs': len(history['train_loss']),
        'model_parameters': model_summary['trainable_parameters']
    }
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    logger.info("=" * 60)
    
    return results


def train_test_split_with_stratify(df: pd.DataFrame, test_size: float, random_state: int, stratify) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets.
    
    Args:
        df: DataFrame to split
        test_size: Fraction for validation
        random_state: Random seed
        stratify: Column to stratify by
    
    Returns:
        Tuple of (train_df, val_df)
    """
    from sklearn.model_selection import train_test_split as sk_split
    return sk_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    from sklearn.model_selection import train_test_split as sk_split
    return sk_split(df, test_size=test_size, random_state=random_state, stratify=stratify)


if __name__ == "__main__":
    results = run_training()
