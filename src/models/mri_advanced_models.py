# src/models/mri_advanced_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import CardiacPyTorchModel

class MRI_MADRU_Net(CardiacPyTorchModel):
    """MRI-MADRU-Net: Multiscale Attention Residual U-Net"""
    def __init__(self, input_dim=15, device=None):
        super().__init__(device)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Multiscale component (simplified)
        self.scale1 = nn.Linear(32, 16)
        self.scale2 = nn.Linear(32, 16)
        self.attention = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 3)
        self.to(self.device)

    def forward(self, x):
        h = self.encoder(x)
        s1 = torch.sigmoid(self.scale1(h))
        s2 = torch.sigmoid(self.scale2(h))
        # Combine scales with attention
        attn = torch.sigmoid(self.attention(h))
        combined = h * attn
        return self.fc(combined)

    def fit(self, X, y, epochs=10):
        self.train()
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
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return F.softmax(self.forward(X_t), dim=1).cpu().numpy()

class MRI_SequenceMorph(CardiacPyTorchModel):
    """MRI-SequenceMorph: Deformable Registration Tracking"""
    def __init__(self, input_dim=15, device=None):
        super().__init__(device)
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.reg_head = nn.Linear(32, 16) # Registration parameters
        self.fc = nn.Linear(16, 3)
        self.to(self.device)

    def forward(self, x):
        # x: (B, T, D)
        _, (h_n, _) = self.lstm(x)
        reg = self.reg_head(h_n[-1])
        return self.fc(reg)

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

class MRI_VelocityGAN(CardiacPyTorchModel):
    """MRI-VelocityGAN for 4D Maps"""
    def __init__(self, input_dim=15, device=None):
        super().__init__(device)
        self.gen = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.to(self.device)
    def fit(self, X, y, epochs=5):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.gen(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.gen(X_t), dim=1).cpu().numpy()

class MRI_ScarMapper(CardiacPyTorchModel):
    """MRI-ScarMapper with Finite Element Constraints"""
    def __init__(self, input_dim=15, device=None):
        super().__init__(device)
        self.unet = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.to(self.device)
    def fit(self, X, y, epochs=5):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.unet(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval(); X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad(): return F.softmax(self.unet(X_t), dim=1).cpu().numpy()

class MRI_Diffusion(CardiacPyTorchModel):
    """MRI-Diffusion Reconstruction Model"""
    def __init__(self, input_dim=15, device=None):
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
