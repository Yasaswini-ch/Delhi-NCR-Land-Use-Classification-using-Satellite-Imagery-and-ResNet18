"""
Q3 Module: CNN Model Definition
Custom CNN architecture for air quality prediction from satellite images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from torchvision import models


class ResNet18LandUse(nn.Module):
    """
    ResNet18 model adapted for land-use classification.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True
    ):
        """
        Initialize ResNet18 model for land-use classification.
        
        Args:
            num_classes: Number of output classes (5 for land-use categories)
            pretrained: Whether to use pretrained weights
        """
        super(ResNet18LandUse, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the final fully connected layer for our number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class AQICNN(nn.Module):
    """
    Convolutional Neural Network for Air Quality Index prediction
    from satellite imagery.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 6,
        conv_layers: List[Dict[str, Any]] = None,
        fc_layers: List[int] = None,
        dropout: float = 0.5
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            conv_layers: List of conv layer configurations
            fc_layers: List of fully connected layer sizes
            dropout: Dropout probability
        """
        super(AQICNN, self).__init__()
        
        if conv_layers is None:
            conv_layers = [
                {'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filters': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filters': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ]
        
        if fc_layers is None:
            fc_layers = [512, 256]
        
        self.conv_layers_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        in_channels = input_channels
        for layer_config in conv_layers:
            out_channels = layer_config['filters']
            kernel_size = layer_config['kernel_size']
            padding = layer_config['padding']
            
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            )
            self.conv_layers_list.append(conv)
            
            bn = nn.BatchNorm2d(out_channels)
            self.batch_norms.append(bn)
            
            in_channels = out_channels
        
        self.flatten = nn.Flatten()
        
        self.fc_layers_list = nn.ModuleList()
        prev_size = 256 * 4 * 4
        
        for fc_size in fc_layers:
            fc = nn.Linear(prev_size, fc_size)
            self.fc_layers_list.append(fc)
            prev_size = fc_size
        
        self.dropout = nn.Dropout(dropout)
        
        self.output = nn.Linear(prev_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        for conv, bn in zip(self.conv_layers_list, self.batch_norms):
            x = F.relu(bn(conv(x)))
            x = self.pool(x)
        
        x = self.flatten(x)
        
        for fc in self.fc_layers_list:
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
        
        Returns:
            List of feature maps from each conv layer
        """
        features = []
        
        for conv, bn in zip(self.conv_layers_list, self.batch_norms):
            x = F.relu(bn(conv(x)))
            x = self.pool(x)
            features.append(x)
        
        return features


class FeatureExtractor(nn.Module):
    """
    Feature extractor using pretrained backbone for transfer learning.
    """
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        """
        Initialize feature extractor.
        
        Args:
            input_channels: Number of input channels
            feature_dim: Dimension of output features
        """
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(256, feature_dim)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int
    ) -> nn.Sequential:
        """
        Create a residual block layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks
        
        Returns:
            Sequential layer
        """
        layers = []
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class AQIRegressor(nn.Module):
    """
    Regression model for direct AQI prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3
    ):
        """
        Initialize regression model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(AQIRegressor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
        
        Returns:
            AQI prediction
        """
        return self.model(x)


class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron for classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: List[int] = None,
        dropout: float = 0.5
    ):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(MLPClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
        
        Returns:
            Class logits
        """
        return self.model(x)


def create_model(config: dict, input_dim: int = None) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        input_dim: Input feature dimension for MLP
    
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'AQICNN')
    
    use_mlp = config.get('training', {}).get('use_mlp', True)
    
    if use_mlp and input_dim is not None:
        model = MLPClassifier(
            input_dim=input_dim,
            num_classes=model_config.get('num_classes', 6),
            hidden_dims=model_config.get('mlp_hidden_dims', [128, 64, 32]),
            dropout=model_config.get('dropout', 0.5)
        )
    elif model_name == 'ResNet18LandUse':
        model = ResNet18LandUse(
            num_classes=model_config.get('num_classes', 5),
            pretrained=model_config.get('pretrained', True)
        )
    else:
        # Default to AQICNN
        model = AQICNN(
            input_channels=model_config.get('input_channels', 3),
            num_classes=model_config.get('num_classes', 6),
            conv_layers=model_config.get('conv_layers'),
            fc_layers=model_config.get('fc_layers'),
            dropout=model_config.get('dropout', 0.5)
        )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple = (3, 64, 64)) -> Dict[str, Any]:
    """
    Get model summary information.
    
    Args:
        model: PyTorch model
        input_size: Input size (channels, height, width)
    
    Returns:
        Dictionary with model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': input_size,
        'num_layers': len(list(model.modules()))
    }
    
    return summary


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    torch.save(checkpoint, path)


def load_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Checkpoint path
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'accuracy': checkpoint['accuracy']
    }
