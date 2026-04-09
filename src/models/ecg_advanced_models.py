# src/models/ecg_advanced_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import CardiacPyTorchModel

class ECG_PINN(CardiacPyTorchModel):
    """ECG-Physics-Informed Neural Network"""
    def __init__(self, input_dim=10, device=None):
        super().__init__(device)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
        self.to(self.device)

    def bidomain_loss(self, x, pred):
        return torch.mean(torch.abs(torch.gradient(pred, dim=1)[0])) * 0.01

    def fit(self, X, y, epochs=5):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.net(X_t)
            loss = F.cross_entropy(logits, y_t) + self.bidomain_loss(X_t, logits)
            loss.backward(); optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.net(X_t), dim=1).cpu().numpy()

class ECG_Hybrid(CardiacPyTorchModel):
    """ECG-BiGRU-BiLSTM-Dilated CNN Hybrid"""
    def __init__(self, input_dim=10, hidden_dim=64, device=None):
        super().__init__(device)
        self.cnn = nn.Conv1d(1, 16, kernel_size=3, padding=1, dilation=2)
        self.gru = nn.GRU(16, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, 3)
        self.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(1); x = F.relu(self.cnn(x)); x = x.permute(0, 2, 1)
        x, _ = self.gru(x); x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

    def fit(self, X, y, epochs=5):
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.forward(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.forward(X_t), dim=1).cpu().numpy()

class ECG_InverseSolver(CardiacPyTorchModel):
    """ECG-Inverse Problem Solver"""
    def __init__(self, input_dim=10, device=None):
        super().__init__(device)
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.to(self.device)
    def fit(self, X, y, epochs=5):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.net(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.net(X_t), dim=1).cpu().numpy()

class ECG_OSACN_Net(CardiacPyTorchModel):
    """ECG-OSACN-Net"""
    def __init__(self, input_dim=10, device=None):
        super().__init__(device)
        self.attn = nn.Linear(input_dim, input_dim)
        self.fc = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        self.to(self.device)
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return self.fc(x * weights)
    def fit(self, X, y, epochs=5):
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.forward(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.forward(X_t), dim=1).cpu().numpy()

class ECG_CNN_LSTM(CardiacPyTorchModel):
    """ECG-CNN-LSTM Multi-Lead Fusion"""
    def __init__(self, input_dim=10, device=None):
        super().__init__(device)
        self.cnn = nn.Conv1d(1, 8, 3, padding=1)
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(160, 3)
        self.to(self.device)
    def forward(self, x):
        x = x.unsqueeze(1); x = F.relu(self.cnn(x)); x = x.permute(0, 2, 1)
        x, _ = self.lstm(x); return self.fc(x.reshape(x.size(0), -1))
    def fit(self, X, y, epochs=5):
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.forward(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.forward(X_t), dim=1).cpu().numpy()
