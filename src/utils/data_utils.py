# src/utils/data_utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def generate_synthetic_timeseries(X, n_timesteps=5, noise_level=0.05):
    """
    Convert static features into synthetic time-series for temporal models.
    X shape: (n_samples, n_features)
    Returns: (n_samples, n_timesteps, n_features)
    """
    n_samples, n_features = X.shape
    series = np.zeros((n_samples, n_timesteps, n_features))
    
    for i in range(n_samples):
        # Base feature vector
        base = X[i]
        for t in range(n_timesteps):
            # Add temporal trend and noise
            trend = (t / n_timesteps) * 0.1 * base
            noise = np.random.normal(0, noise_level, n_features)
            series[i, t] = base + trend + noise
            
    return series

def prepare_ehr_features(X):
    """
    Enhance EHR features with temporal metadata for GBT.
    """
    df = pd.DataFrame(X)
    # Simulate rate of change features (placeholder logic)
    df['trend_risk'] = df.mean(axis=1) * 0.1
    df['variability'] = df.std(axis=1) * 0.05
    return df.values

def to_tensor(X, device='cpu'):
    if isinstance(X, pd.DataFrame):
        X = X.values
    return torch.FloatTensor(X).to(device)

def get_dataloader(X, y, batch_size=32, shuffle=True):
    from torch.utils.data import TensorDataset, DataLoader
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
