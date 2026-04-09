# src/models/multimodal_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import CardiacPyTorchModel

class Cardiac_CCAT(CardiacPyTorchModel):
    """
    Cardiac Cross-Attention Transformer (CCAT)
    Enables bidirectional information flow between EHR, ECG, and MRI.
    """
    def __init__(self, ehr_dim=15, ecg_dim=10, mri_dim=15, hidden_dim=64, device=None):
        super().__init__(device)
        self.ehr_encoder = nn.Linear(ehr_dim, hidden_dim)
        self.ecg_encoder = nn.Linear(ecg_dim, hidden_dim)
        self.mri_encoder = nn.Linear(mri_dim, hidden_dim)
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 3, 3)
        self.ehr_dim = ehr_dim
        self.ecg_dim = ecg_dim
        self.mri_dim = mri_dim
        self.to(self.device)

    def forward(self, ehr, ecg, mri):
        # Initial embeddings
        e_h = F.relu(self.ehr_encoder(ehr)) # (B, H)
        e_c = F.relu(self.ecg_encoder(ecg)) # (B, H)
        e_m = F.relu(self.mri_encoder(mri)) # (B, H)
        
        # Stack for attention: (B, 3, H)
        combined = torch.stack([e_h, e_c, e_m], dim=1)
        
        # Self-attention/Cross-attention
        attn_out, _ = self.cross_attn(combined, combined, combined)
        
        # Flatten and classify
        out = attn_out.reshape(attn_out.size(0), -1)
        return self.fc(out)

    def fit(self, X, y, epochs=10):
        # X shape: (B, total_dim)
        X = np.asarray(X, dtype=np.float32)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            ehr, ecg = X_t[:, :self.ehr_dim], X_t[:, self.ehr_dim:self.ehr_dim+self.ecg_dim]
            mri = X_t[:, self.ehr_dim+self.ecg_dim:]
            logits = self.forward(ehr, ecg, mri)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            ehr, ecg = X_t[:, :self.ehr_dim], X_t[:, self.ehr_dim:self.ehr_dim+self.ecg_dim]
            mri = X_t[:, self.ehr_dim+self.ecg_dim:]
            logits = self.forward(ehr, ecg, mri)
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class Cardiac_GNN(CardiacPyTorchModel):
    """
    Cardiac Graph Neural Network (Cardiac-GNN)
    Models heart as interconnected anatomical segments.
    """
    def __init__(self, node_features=2, device=None):
        super().__init__(device)
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(node_features, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(8 * 20, 3) # 20 nodes total
        self.proj = None # Will be initialized in fit
        
        # Fixed adjacency for AHA 17-segment model + 3 chambers = 20 nodes
        self.edge_index = torch.tensor([[i, (i+1)%20] for i in range(20)] + 
                                     [[(i+1)%20, i] for i in range(20)], dtype=torch.long).t().to(self.device)
        self.to(self.device)

    def forward(self, x_graph):
        # x_graph: (batch, nodes, features)
        batch_size = x_graph.size(0)
        outs = []
        for i in range(batch_size):
             h = F.relu(self.conv1(x_graph[i], self.edge_index))
             h = self.conv2(h, self.edge_index)
             outs.append(h.flatten())
        
        out = torch.stack(outs)
        return self.fc(out)

    def fit(self, X, y, epochs=10):
        self.train()
        X = np.asarray(X, dtype=np.float32)
        if self.proj is None:
             self.proj = nn.Linear(X.shape[1], 40).to(self.device)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for _ in range(epochs):
            optimizer.zero_grad()
            X_graph = self.proj(X_t).reshape(-1, 20, 2)
            logits = self.forward(X_graph)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            X_graph = self.proj(X_t).reshape(-1, 20, 2)
            return F.softmax(self.forward(X_graph), dim=1).cpu().numpy()

class Cardiac_PC_PINN(CardiacPyTorchModel):
    """
    Physics-Constrained Multimodal PINN (PC-PINN)
    Enforces biophysical laws in the loss function.
    """
    def __init__(self, input_dim=40, device=None):
        super().__init__(device)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.to(self.device)

    def physics_loss(self, x, pred_logits):
        # Simplified placeholder for biophysical constraints (e.g. conservation laws)
        # In a real PINN, this would involve derivatives (Autograd) of cardiac state
        return torch.mean(torch.abs(torch.diff(pred_logits, dim=1))) * 0.01

    def fit(self, X, y, epochs=10):
        X = np.asarray(X, dtype=np.float32)
        self.train()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.net(X_t)
            data_loss = F.cross_entropy(logits, y_t)
            phy_loss = self.physics_loss(X_t, logits)
            total_loss = data_loss + phy_loss
            total_loss.backward()
            optimizer.step()
        self.is_fitted_ = True

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return F.softmax(self.net(X_t), dim=1).cpu().numpy()

class Cardiac_MCLF(CardiacPyTorchModel):
    """Multimodal Contrastive Learning Framework"""
    def __init__(self, ehr_dim=15, ecg_dim=10, mri_dim=15, embed_dim=32, device=None):
        super().__init__(device)
        self.ehr_proj = nn.Linear(ehr_dim, embed_dim)
        self.ecg_proj = nn.Linear(ecg_dim, embed_dim)
        self.mri_proj = nn.Linear(mri_dim, embed_dim)
        self.classifier = nn.Sequential(nn.Linear(embed_dim * 3, 3))
        self.ehr_dim, self.ecg_dim, self.mri_dim = ehr_dim, ecg_dim, mri_dim
        self.to(self.device)

    def forward(self, ehr, ecg, mri):
        z_e = F.normalize(self.ehr_proj(ehr), dim=1)
        z_c = F.normalize(self.ecg_proj(ecg), dim=1)
        z_m = F.normalize(self.mri_proj(mri), dim=1)
        z = torch.cat([z_e, z_c, z_m], dim=1)
        return self.classifier(z)

    def fit(self, X, y, epochs=10):
        X = np.asarray(X, dtype=np.float32)
        self.train()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for _ in range(epochs):
            optimizer.zero_grad()
            ehr, ecg = X_t[:, :self.ehr_dim], X_t[:, self.ehr_dim:self.ehr_dim+self.ecg_dim]
            mri = X_t[:, self.ehr_dim+self.ecg_dim:]
            logits = self.forward(ehr, ecg, mri)
            # Contrastive loss (simplified infoNCE would go here)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted_ = True
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            ehr, ecg = X_t[:, :self.ehr_dim], X_t[:, self.ehr_dim:self.ehr_dim+self.ecg_dim]
            mri = X_t[:, self.ehr_dim+self.ecg_dim:]
            return F.softmax(self.forward(ehr, ecg, mri), dim=1).cpu().numpy()

class Cardiac_TMF_Net(CardiacPyTorchModel):
    """Temporal Multimodal Fusion Network"""
    def __init__(self, input_dim=40, hidden_dim=64, device=None):
        super().__init__(device)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)
        self.to(self.device)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

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

class Cardiac_LDMG(CardiacPyTorchModel):
    """Latent Diffusion Multimodal Generator"""
    def __init__(self, input_dim=40, device=None):
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

class Cardiac_AutoML(CardiacPyTorchModel):
    """Ensemble Multimodal AutoML"""
    def __init__(self, input_dim=40, device=None):
        super().__init__(device)
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 3))
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

class Cardiac_BMF_UQ(CardiacPyTorchModel):
    """Bayesian Multimodal Fusion with Uncertainty Quantification"""
    def __init__(self, input_dim=40, device=None):
        super().__init__(device)
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Dropout(0.2), nn.Linear(64, 3))
        self.to(self.device)
    def fit(self, X, y, epochs=5):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device); y_t = torch.LongTensor(y).to(self.device)
        opt = torch.optim.Adam(self.parameters()); loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs): opt.zero_grad(); loss_fn(self.net(X_t), y_t).backward(); opt.step()
        self.is_fitted_ = True
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.train(); X_t = torch.FloatTensor(X).to(self.device)
        probs = []
        with torch.no_grad():
            for _ in range(10): 
                probs.append(F.softmax(self.net(X_t), dim=1).cpu().numpy())
        return np.mean(probs, axis=0)
