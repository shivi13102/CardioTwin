# src/utils/helpers.py
"""
Utility helper functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import pickle
import joblib
from pathlib import Path

def save_results(results_dict, filepath):
    """Save results dictionary to file."""
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4, default=str)

def load_results(filepath):
    """Load results from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_risk_categories(risk_scores):
    """Create risk categories from continuous scores."""
    categories = []
    for score in risk_scores:
        if score < 0.33:
            categories.append('low_risk')
        elif score < 0.67:
            categories.append('medium_risk')
        else:
            categories.append('high_risk')
    return categories

def calculate_risk_burden(clinical_score, ecg_score, mri_score, weights=None):
    """Calculate combined risk burden."""
    if weights is None:
        weights = {'clinical': 0.3, 'ecg': 0.3, 'mri': 0.4}
    
    total_burden = (clinical_score * weights['clinical'] + 
                   ecg_score * weights['ecg'] + 
                   mri_score * weights['mri'])
    
    return total_burden

def create_synthetic_patient_id(modality_indices):
    """Create synthetic patient ID from modality indices."""
    return f"SYN_{modality_indices['ehr']}_{modality_indices['ecg']}_{modality_indices['mri']}"

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Evaluate predictions with multiple metrics."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score)
    
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    return results

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_classification_report(y_true, y_pred, target_names):
    """Print classification report."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

def save_model(model, filepath):
    """Save model to file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load model from file."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def ensure_directory_exists(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_feature_statistics(df):
    """Get basic statistics for features."""
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numerical_stats': df.describe().to_dict(),
        'categorical_stats': {
            col: df[col].value_counts().to_dict() 
            for col in df.select_dtypes(include=['object']).columns
        }
    }
    return stats

def normalize_features(X, method='zscore'):
    """Normalize features using specified method."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

def handle_imbalance(X, y, method='smote'):
    """Handle class imbalance using specified method."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'under':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError(f"Unknown imbalance handling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def create_correlation_heatmap(df, save_path=None):
    """Create correlation heatmap for numerical features."""
    numerical_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    corr_matrix = numerical_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_model_comparison(results_dict):
    """Print comparison of multiple models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_name, metrics in results_dict.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric != 'model':
                print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*80)

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for predictions."""
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - ci, mean + ci

def bootstrap_evaluation(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Bootstrap evaluation for robust metrics."""
    np.random.seed(random_state)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    bootstrap_results = {metric: [] for metric in metrics}
    
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        bootstrap_results['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrap_results['precision'].append(precision_score(y_true_boot, y_pred_boot, average='weighted'))
        bootstrap_results['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted'))
        bootstrap_results['f1'].append(f1_score(y_true_boot, y_pred_boot, average='weighted'))
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_results.items():
        mean, lower, upper = calculate_confidence_interval(values)
        confidence_intervals[metric] = {
            'mean': mean,
            'lower_ci': lower,
            'upper_ci': upper
        }
    
    return confidence_intervals