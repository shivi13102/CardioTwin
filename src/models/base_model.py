# src/models/base_model.py
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import joblib
import os

class CardiacModel(BaseEstimator, ClassifierMixin):
    """Base class for all cardiac models to ensure sklearn compatibility."""
    
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = [0, 1, 2] # low, medium, high risk
        self.is_fitted_ = False
        
    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        raise NotImplementedError("Subclasses must implement predict_proba()")
        
    def save(self, path):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if hasattr(self, 'model') and isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self, path)
            
    def load(self, path):
        """Load model from disk."""
        if hasattr(self, 'model') and isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            loaded = joblib.load(path)
            self.__dict__.update(loaded.__dict__)
        return self

class CardiacPyTorchModel(CardiacModel, nn.Module):
    """Base class for PyTorch-based cardiac models."""
    
    def __init__(self, device=None):
        nn.Module.__init__(self)
        CardiacModel.__init__(self, device)
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
