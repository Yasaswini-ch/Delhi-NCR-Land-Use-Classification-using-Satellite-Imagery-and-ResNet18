"""
Evaluation module for trained CNN model.
Computes metrics, confusion matrix, cross-validation, baseline comparison, and visualizations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier, MLPRegressor as SklearnMLPRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, load_config
from src.q3_model import AQICNN, create_model, MLPClassifier


SIX_CLASS_NAMES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
THREE_CLASS_NAMES = ['Low', 'Medium', 'High']


def map_to_three_class(labels: np.ndarray) -> np.ndarray:
    """
    Map 6-class labels to 3-class (Low/Medium/High).
    
    Args:
        labels: 6-class labels (0-5)
    
    Returns:
        3-class labels (0=Low, 1=Medium, 2=High)
    """
    three_class = np.zeros_like(labels)
    three_class[labels >= 3] = 2  # Poor, Very Poor, Severe -> High
    three_class[labels == 2] = 1  # Moderate -> Medium
    # Good, Satisfactory -> Low (already 0)
    return three_class


def get_class_names(config: dict) -> List[str]:
    """Get class names based on classification mode."""
    if config['task'].get('classification_mode', '6_class') == '3_class':
        return THREE_CLASS_NAMES
    return SIX_CLASS_NAMES


def get_num_classes(config: dict) -> int:
    """Get number of classes based on classification mode."""
    if config['task'].get('classification_mode', '6_class') == '3_class':
        return 3
    return 6


def load_all_data(config: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all processed data for evaluation.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (features, labels, feature_names)
    """
    logger = setup_logger("Evaluate")
    
    processed_path = Path(config['paths']['processed_data'])
    train_df = pd.read_csv(processed_path / 'train.csv')
    test_df = pd.read_csv(processed_path / 'test.csv')
    
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    exclude_cols = ['station_id', 'station_name', 'date', 'aqi', 'aqi_category', 
                    'pm25', 'pm10', 'no2', 'o3', 'label', 'longitude_x', 'latitude_x', 
                    'longitude_y', 'latitude_y', 'image_name']
    
    available_cols = [c for c in all_df.columns 
                      if c not in exclude_cols 
                      and all_df[c].dtype in ['int64', 'float64', 'float32']]
    
    X = all_df[available_cols].values
    y = all_df['label'].values
    
    X = np.nan_to_num(X, nan=0.0)
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y, available_cols


def load_test_data(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data for evaluation.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (features, labels)
    """
    logger = setup_logger("Evaluate")
    
    processed_path = Path(config['paths']['processed_data'])
    test_df = pd.read_csv(processed_path / 'test.csv')
    
    exclude_cols = ['station_id', 'station_name', 'date', 'aqi', 'aqi_category', 
                   'pm25', 'pm10', 'no2', 'o3', 'label', 'longitude_x', 'latitude_x', 
                   'longitude_y', 'latitude_y', 'image_name']
    
    available_cols = [c for c in test_df.columns 
                     if c not in exclude_cols 
                     and test_df[c].dtype in ['int64', 'float64', 'float32']]
    
    X_test = test_df[available_cols].values
    y_test = test_df['label'].values
    
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    X_test = X_test.astype(np.float32)
    
    logger.info(f"Loaded test data: {X_test.shape[0]} samples")
    
    return X_test, y_test


def load_trained_model(config: dict, device: torch.device) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        config: Configuration dictionary
        device: Computing device
    
    Returns:
        Loaded model
    """
    logger = setup_logger("Evaluate")
    
    from src.q3_model import MLPClassifier
    import pandas as pd
    
    processed_path = Path(config['paths']['processed_data'])
    test_df = pd.read_csv(processed_path / 'test.csv')
    
    exclude_cols = ['station_id', 'station_name', 'date', 'aqi', 'aqi_category', 
                    'pm25', 'pm10', 'no2', 'o3', 'label', 'longitude_x', 'latitude_x', 
                    'longitude_y', 'latitude_y', 'image_name']
    
    available_cols = [c for c in test_df.columns 
                      if c not in exclude_cols 
                      and test_df[c].dtype in ['int64', 'float64', 'float32']]
    num_features = len(available_cols)
    
    model_config = config.get('model', {})
    config_num_classes = model_config.get('num_classes', 6)
    num_classes = get_num_classes(config)
    
    model = MLPClassifier(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dims=model_config.get('mlp_hidden_dims', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.5)
    )
    
    output_path = Path(config['paths']['outputs'])
    model_path = output_path / 'best_model.pth'
    
    if model_path.exists():
        if config_num_classes == num_classes:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from: {model_path}")
        else:
            logger.warning(f"Model was trained for {config_num_classes} classes but current config requires {num_classes} classes. Using untrained model.")
    else:
        logger.warning(f"Model file not found: {model_path}, using untrained model")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict(
    model: nn.Module,
    X_test: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        device: Computing device
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = outputs.max(1)
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        config: Configuration dictionary
    
    Returns:
        Dictionary with metrics
    """
    logger = setup_logger("Evaluate")
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    prec_recall = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision = prec_recall[0]
    recall = prec_recall[1]
    support = prec_recall[2]
    
    num_classes = get_num_classes(config)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_micro': float(f1_micro),
        'precision_per_class': precision.tolist() if hasattr(precision, 'tolist') else list(precision),
        'recall_per_class': recall.tolist() if hasattr(recall, 'tolist') else list(recall),
        'support_per_class': support.tolist() if hasattr(support, 'tolist') else list(support)
    }
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        config: Configuration dictionary
    
    Returns:
        Classification report string
    """
    logger = setup_logger("Evaluate")
    
    class_names = get_class_names(config)
    num_classes = get_num_classes(config)
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    labels_to_use = list(range(min(unique_labels), max(unique_labels) + 1))
    
    report = classification_report(
        y_true, y_pred,
        labels=labels_to_use,
        target_names=[class_names[i] for i in labels_to_use],
        zero_division=0
    )
    
    logger.info("\n" + report)
    
    return report


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    model_type: str = 'pytorch_mlp'
) -> Dict[str, float]:
    """
    Run stratified k-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        config: Configuration dictionary
        model_type: Type of model ('pytorch_mlp', 'logistic_regression', 'random_forest', 'mlp')
    
    Returns:
        Dictionary with CV results (mean ± std)
    """
    logger = setup_logger("Evaluate")
    
    n_folds = config['dataset'].get('cross_validation_folds', 5)
    seed = config['dataset']['random_seed']
    
    logger.info(f"Running {n_folds}-fold stratified cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_weighteds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        if model_type == 'pytorch_mlp':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_classes = get_num_classes(config)
            model = MLPClassifier(
                input_dim=X.shape[1],
                num_classes=num_classes,
                hidden_dims=config['model'].get('mlp_hidden_dims', [128, 64, 32]),
                dropout=config['model'].get('dropout', 0.5)
            ).to(device)
            
            X_train_t = torch.FloatTensor(X_train_fold).to(device)
            y_train_t = torch.LongTensor(y_train_fold).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val_fold).to(device)
                outputs = model(X_val_t)
                _, y_pred_fold = outputs.max(1)
                y_pred_fold = y_pred_fold.cpu().numpy()
                
        elif model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
            
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=seed)
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
            
        elif model_type == 'sklearn_mlp':
            model = SklearnMLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=seed
            )
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
        
        fold_acc = accuracy_score(y_val_fold, y_pred_fold)
        fold_f1_macro = f1_score(y_val_fold, y_pred_fold, average='macro')
        fold_f1_weighted = f1_score(y_val_fold, y_pred_fold, average='weighted')
        
        fold_accuracies.append(fold_acc)
        fold_f1_macros.append(fold_f1_macro)
        fold_f1_weighteds.append(fold_f1_weighted)
        
        logger.info(f"Fold {fold+1}: Acc={fold_acc:.4f}, F1-macro={fold_f1_macro:.4f}")
    
    results = {
        f'{model_type}_accuracy_mean': np.mean(fold_accuracies),
        f'{model_type}_accuracy_std': np.std(fold_accuracies),
        f'{model_type}_f1_macro_mean': np.mean(fold_f1_macros),
        f'{model_type}_f1_macro_std': np.std(fold_f1_macros),
        f'{model_type}_f1_weighted_mean': np.mean(fold_f1_weighteds),
        f'{model_type}_f1_weighted_std': np.std(fold_f1_weighteds)
    }
    
    logger.info(f"\n{model_type.upper()} CV Results:")
    logger.info(f"Accuracy: {results[f'{model_type}_accuracy_mean']:.4f} ± {results[f'{model_type}_accuracy_std']:.4f}")
    logger.info(f"F1-macro: {results[f'{model_type}_f1_macro_mean']:.4f} ± {results[f'{model_type}_f1_macro_std']:.4f}")
    
    return results


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    model_type: str = 'random_forest'
) -> Dict[str, float]:
    """
    Run k-fold cross-validation for regression.
    
    Args:
        X: Features
        y: Target values (AQI)
        config: Configuration dictionary
        model_type: Type of model ('random_forest', 'mlp_regressor')
    
    Returns:
        Dictionary with CV results (mean ± std)
    """
    logger = setup_logger("Evaluate")
    
    n_folds = config['dataset'].get('cross_validation_folds', 5)
    seed = config['dataset']['random_seed']
    
    logger.info(f"Running {n_folds}-fold cross-validation for regression...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_rmses = []
    fold_maes = []
    fold_r2s = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=seed)
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
            
        elif model_type == 'mlp_regressor':
            model = SklearnMLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=seed
            )
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
        
        metrics = compute_regression_metrics(y_val_fold, y_pred_fold)
        
        fold_rmses.append(metrics['rmse'])
        fold_maes.append(metrics['mae'])
        fold_r2s.append(metrics['r2'])
        
        logger.info(f"Fold {fold+1}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
    
    results = {
        f'{model_type}_rmse_mean': np.mean(fold_rmses),
        f'{model_type}_rmse_std': np.std(fold_rmses),
        f'{model_type}_mae_mean': np.mean(fold_maes),
        f'{model_type}_mae_std': np.std(fold_maes),
        f'{model_type}_r2_mean': np.mean(fold_r2s),
        f'{model_type}_r2_std': np.std(fold_r2s)
    }
    
    logger.info(f"\n{model_type.upper()} Regression CV Results:")
    logger.info(f"RMSE: {results[f'{model_type}_rmse_mean']:.2f} ± {results[f'{model_type}_rmse_std']:.2f}")
    logger.info(f"MAE: {results[f'{model_type}_mae_mean']:.2f} ± {results[f'{model_type}_mae_std']:.2f}")
    logger.info(f"R²: {results[f'{model_type}_r2_mean']:.4f} ± {results[f'{model_type}_r2_std']:.4f}")
    
    return results


def run_baseline_comparison(
    X: np.ndarray,
    y: np.ndarray,
    config: dict
) -> pd.DataFrame:
    """
    Run baseline model comparison with cross-validation.
    
    Args:
        X: Features
        y: Labels
        config: Configuration dictionary
    
    Returns:
        DataFrame with comparison results
    """
    logger = setup_logger("Evaluate")
    logger.info("=" * 60)
    logger.info("Running Baseline Model Comparison")
    logger.info("=" * 60)
    
    task_type = config['task'].get('task_type', 'classification')
    results_list = []
    
    if task_type == 'classification':
        models = config.get('baseline_models', ['logistic_regression', 'random_forest', 'mlp'])
        
        for model_name in models:
            logger.info(f"\n--- {model_name} ---")
            
            cv_key = model_name
            if model_name == 'logistic_regression':
                cv_results = run_cross_validation(X, y, config, 'logistic_regression')
            elif model_name == 'random_forest':
                cv_results = run_cross_validation(X, y, config, 'random_forest')
            elif model_name == 'mlp':
                cv_results = run_cross_validation(X, y, config, 'sklearn_mlp')
                cv_key = 'sklearn_mlp'
            elif model_name == 'pytorch_mlp':
                cv_results = run_cross_validation(X, y, config, 'pytorch_mlp')
            
            results_list.append({
                'model': model_name,
                'accuracy_mean': cv_results.get(f'{cv_key}_accuracy_mean', 0),
                'accuracy_std': cv_results.get(f'{cv_key}_accuracy_std', 0),
                'f1_macro_mean': cv_results.get(f'{cv_key}_f1_macro_mean', 0),
                'f1_macro_std': cv_results.get(f'{cv_key}_f1_macro_std', 0)
            })
    
    else:
        logger.info("\n--- Random Forest Regressor ---")
        rf_results = run_regression_cv(X, y, config, 'random_forest')
        results_list.append({
            'model': 'random_forest',
            'rmse_mean': rf_results['random_forest_rmse_mean'],
            'rmse_std': rf_results['random_forest_rmse_std'],
            'mae_mean': rf_results['random_forest_mae_mean'],
            'mae_std': rf_results['random_forest_mae_std'],
            'r2_mean': rf_results['random_forest_r2_mean'],
            'r2_std': rf_results['random_forest_r2_std']
        })
        
        logger.info("\n--- MLP Regressor ---")
        mlp_results = run_regression_cv(X, y, config, 'mlp_regressor')
        results_list.append({
            'model': 'mlp_regressor',
            'rmse_mean': mlp_results['mlp_regressor_rmse_mean'],
            'rmse_std': mlp_results['mlp_regressor_rmse_std'],
            'mae_mean': mlp_results['mlp_regressor_mae_mean'],
            'mae_std': mlp_results['mlp_regressor_mae_std'],
            'r2_mean': mlp_results['mlp_regressor_r2_mean'],
            'r2_std': mlp_results['mlp_regressor_r2_std']
        })
    
    results_df = pd.DataFrame(results_list)
    
    output_path = Path(config['paths']['outputs'])
    save_path = output_path / 'baseline_comparison.csv'
    results_df.to_csv(save_path, index=False)
    logger.info(f"\nBaseline comparison saved: {save_path}")
    
    return results_df


def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: dict
) -> Dict[str, float]:
    """
    Compute feature importance using Random Forest.
    
    Args:
        X: Features
        y: Labels
        feature_names: List of feature names
        config: Configuration dictionary
    
    Returns:
        Dictionary with feature importance scores
    """
    logger = setup_logger("Evaluate")
    
    seed = config['dataset']['random_seed']
    
    logger.info("Computing feature importance using Random Forest...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_scaled, y)
    
    importances = rf.feature_importances_
    
    importance_dict = dict(zip(feature_names, importances))
    
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nTop 10 Feature Importances:")
    for feat, imp in sorted_importance[:10]:
        logger.info(f"  {feat}: {imp:.4f}")
    
    return importance_dict


def plot_feature_importance(
    importance_dict: Dict[str, float],
    config: dict
) -> None:
    """
    Plot and save feature importance.
    
    Args:
        importance_dict: Dictionary with feature importance scores
        config: Configuration dictionary
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_importance]
    importances = [x[1] for x in sorted_importance]
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    
    plt.barh(range(len(features)), importances, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path = output_path / 'feature_importance.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict,
    normalize: bool = True
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        config: Configuration dictionary
        normalize: Whether to normalize the confusion matrix
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    class_names = get_class_names(config)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = '(Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Air Quality Classification {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    suffix = '_normalized' if normalize else ''
    save_path = output_path / f'confusion_matrix{suffix}.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict
) -> None:
    """
    Plot true vs predicted class distribution.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        config: Configuration dictionary
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    class_names = get_class_names(config)
    num_classes = get_num_classes(config)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    true_counts = true_counts.reindex(range(num_classes), fill_value=0)
    pred_counts = pred_counts.reindex(range(num_classes), fill_value=0)
    
    axes[0].bar(class_names, true_counts.values, color='blue', alpha=0.7)
    axes[0].set_title('True Label Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('AQI Category')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(class_names, pred_counts.values, color='green', alpha=0.7)
    axes[1].set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('AQI Category')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    save_path = output_path / 'prediction_distribution.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction distribution saved: {save_path}")


def plot_per_class_metrics(
    metrics: Dict[str, float],
    config: dict,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> None:
    """
    Plot per-class precision and recall.
    
    Args:
        metrics: Metrics dictionary
        config: Configuration dictionary
        y_true: True labels (optional, for computing per-class metrics)
        y_pred: Predicted labels (optional)
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    class_names = get_class_names(config)
    num_classes = get_num_classes(config)
    
    if y_true is not None and y_pred is not None:
        all_labels = np.arange(num_classes)
        prec_recall = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        precision = np.array(prec_recall[0])
        recall = np.array(prec_recall[1])
    else:
        precision = metrics.get('precision_per_class', [])
        recall = metrics.get('recall_per_class', [])
        if not precision or not recall:
            logger.warning("No precision/recall data available for plotting")
            return
        precision = np.array(precision)
        recall = np.array(recall)
    
    x = np.arange(num_classes)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    line1, = ax.plot(x, precision, 'o-', label='Precision', color='blue', linewidth=2, markersize=8)
    line2, = ax.plot(x, recall, 's-', label='Recall', color='orange', linewidth=2, markersize=8)
    
    ax.set_xlabel('AQI Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Precision and Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    for i in range(num_classes):
        p = precision[i] if i < len(precision) else 0
        r = recall[i] if i < len(recall) else 0
        if p > 0:
            ax.annotate(f'{p:.2f}', (i, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        if r > 0:
            ax.annotate(f'{r:.2f}', (i, r), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.tight_layout()
    
    save_path = output_path / 'per_class_metrics.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-class metrics saved: {save_path}")


def plot_baseline_comparison(results_df: pd.DataFrame, config: dict) -> None:
    """
    Plot baseline model comparison.
    
    Args:
        results_df: DataFrame with comparison results
        config: Configuration dictionary
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    task_type = config['task'].get('task_type', 'classification')
    
    plt.figure(figsize=(10, 6))
    
    if task_type == 'classification':
        models = results_df['model'].values
        x = np.arange(len(models))
        width = 0.35
        
        acc_means = results_df['accuracy_mean'].values
        acc_stds = results_df['accuracy_std'].values
        f1_means = results_df['f1_macro_mean'].values
        f1_stds = results_df['f1_macro_std'].values
        
        plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy', color='steelblue', alpha=0.8, capsize=5)
        plt.bar(x + width/2, f1_means, width, yerr=f1_stds, label='F1-Macro', color='coral', alpha=0.8, capsize=5)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Baseline Model Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
        plt.xticks(x, models, rotation=15)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
    else:
        models = results_df['model'].values
        x = np.arange(len(models))
        width = 0.25
        
        rmse_means = results_df['rmse_mean'].values
        mae_means = results_df['mae_mean'].values
        
        plt.bar(x - width/2, rmse_means, width, label='RMSE', color='steelblue', alpha=0.8)
        plt.bar(x + width/2, mae_means, width, label='MAE', color='coral', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title('Baseline Regression Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_path / 'baseline_comparison.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Baseline comparison plot saved: {save_path}")


def save_metrics(metrics: Dict[str, float], config: dict) -> None:
    """
    Save metrics to file.
    
    Args:
        metrics: Metrics dictionary
        config: Configuration dictionary
    """
    logger = setup_logger("Evaluate")
    output_path = Path(config['paths']['outputs'])
    
    metrics_df = pd.DataFrame([metrics])
    save_path = output_path / 'evaluation_metrics.csv'
    metrics_df.to_csv(save_path, index=False)
    
    logger.info(f"Metrics saved: {save_path}")


def run_evaluation(config_path: Optional[str] = None) -> dict:
    """
    Main evaluation function.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Dictionary with evaluation results
    """
    logger = setup_logger("Evaluate")
    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)
    
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / 'configs' / 'config.yaml')
    
    config = load_config(config_path)
    
    task_type = config['task'].get('task_type', 'classification')
    classification_mode = config['task'].get('classification_mode', '6_class')
    
    logger.info(f"Task type: {task_type}")
    logger.info(f"Classification mode: {classification_mode}")
    
    seed = config['dataset']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    X, y, feature_names = load_all_data(config)
    
    if classification_mode == '3_class':
        logger.info("Converting to 3-class labels (Low/Medium/High)")
        y = map_to_three_class(y)
    
    if task_type == 'regression':
        processed_path = Path(config['paths']['processed_data'])
        train_df = pd.read_csv(processed_path / 'train.csv')
        test_df = pd.read_csv(processed_path / 'test.csv')
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        y = all_df['aqi'].values
    
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    baseline_df = run_baseline_comparison(X, y, config)
    plot_baseline_comparison(baseline_df, config)
    
    if task_type == 'classification':
        importance_dict = compute_feature_importance(X, y, feature_names, config)
        plot_feature_importance(importance_dict, config)
    
    X_test, y_test = load_test_data(config)
    
    if classification_mode == '3_class':
        y_test = map_to_three_class(y_test)
    
    model = load_trained_model(config, device)
    
    y_pred, y_prob = predict(model, X_test, device)
    
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, config=config)
    
    report = generate_classification_report(y_test, y_pred, config)
    
    if config['output'].get('save_confusion_matrix', True):
        plot_confusion_matrix(y_test, y_pred, config, normalize=False)
        plot_confusion_matrix(y_test, y_pred, config, normalize=True)
        plot_class_distribution(y_test, y_pred, config)
        plot_per_class_metrics(metrics, config, y_test, y_pred)
    
    save_metrics(metrics, config)
    
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    results = run_evaluation()
