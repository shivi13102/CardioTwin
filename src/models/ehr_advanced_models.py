# src/models/ehr_advanced_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import CardiacModel, CardiacPyTorchModel
# from .unimodal_models import UnimodalModels # Removed to avoid circularity

class EHR_TFT(CardiacPyTorchModel):
    """
    EHR-Temporal Fusion Transformer (EHR-TFT)
    Simplified version focusing on attention and variable selection.
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, device=None):
        super().__init__(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Variable Selection Network (simplified)
        self.vsn = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Output layers for 3 classes
        self.fc = nn.Linear(hidden_dim, 2)
        self.to(self.device)

    def forward(self, x):
        # x shape: (batch, n_timesteps, input_dim)
        batch_size, timesteps, _ = x.size()
        
        # VSN
        x = F.relu(self.vsn(x))
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        
        # Global average pooling over time
        pooled = attn_out.mean(dim=1)
        
        logits = self.fc(pooled)
        return logits

    def fit(self, X, y, epochs=10, batch_size=32):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare data (ensure 3D and float32)
        if len(X.shape) == 2:
             from src.utils.data_utils import generate_synthetic_timeseries
             X = generate_synthetic_timeseries(X)
        
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        self.eval()
        if len(X.shape) == 2:
             from src.utils.data_utils import generate_synthetic_timeseries
             X = generate_synthetic_timeseries(X)
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class EHR_BayesNN(CardiacPyTorchModel):
    """
    EHR-Bayesian Neural Network (EHR-BayesNN)
    Uses MC Dropout for uncertainty estimation.
    """
    def __init__(self, input_dim, hidden_dim=128, device=None):
        super().__init__(device)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def fit(self, X, y, epochs=15):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X, mc_samples=20):
        # Keep dropout active during inference for MC sampling
        self.train() 
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.FloatTensor(X).to(self.device)
        probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(X_t)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.mean(probs, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class EHR_LSTM_Attention(CardiacPyTorchModel):
    def __init__(self, input_dim, hidden_dim=64, device=None):
        super().__init__(device)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.to(self.device)

    def forward(self, x):
        # x: (batch, seq, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)

    def fit(self, X, y, epochs=10):
        self.train()
        if len(X.shape) == 2:
             from src.utils.data_utils import generate_synthetic_timeseries
             X = generate_synthetic_timeseries(X)
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        self.eval()
        if len(X.shape) == 2:
             from src.utils.data_utils import generate_synthetic_timeseries
             X = generate_synthetic_timeseries(X)
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return F.softmax(self.forward(X_t), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class EHR_GBT_Time(CardiacModel):
    """EHR-Gradient Boosting with Engineered Time Features"""
    def __init__(self, device=None):
        super().__init__(device)
        from xgboost import XGBClassifier
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    def fit(self, X, y):
        # In a real scenario, this would use the time-series preparation logic
        self.model.fit(X, y)
        self.is_fitted_ = True
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def predict(self, X):
        return self.model.predict(X)
