import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import os
import sys

# Add root directory to path to import config and src
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from src.models.cardiac_fm import CardiacFM
from src.fusion.clinical_aligner import SyntheticFusion
from src.preprocessing.ehr_preprocessor import EHRPreprocessor
from src.feature_extraction.feature_extractor import FeatureExtractor

class MultimodalDataset(Dataset):
    def __init__(self, ehr, ecg, mri, risk, progression):
        self.ehr = torch.FloatTensor(ehr)
        self.ecg = torch.FloatTensor(ecg)
        self.mri = torch.FloatTensor(mri)
        self.risk = torch.LongTensor(risk)
        prog_tensor = torch.FloatTensor(progression)
        if len(prog_tensor.shape) == 1:
            self.progression = prog_tensor.view(-1, 1)
        else:
            self.progression = prog_tensor

    def __len__(self):
        return len(self.ehr)

    def __getitem__(self, idx):
        return self.ehr[idx], self.ecg[idx], self.mri[idx], self.risk[idx], self.progression[idx]

def train_model():
    config = Config()
    
    # 1. Load or Generate Fused Data
    fused_path = config.FUSED_DATA_DIR / 'fused_multimodal_dataset.csv'
    
    if not fused_path.exists():
        print("Fused dataset not found. Generating synthetic fusion...")
        # (Simplified generation logic for prototype readiness)
        # In a real scenario, this would call the full pipeline.
        # For now, we'll assume the user has run main.py or we can mock it here.
        # Let's check for ehr_processed first.
        try:
             ehr_processed = joblib.load(config.PROCESSED_DATA_DIR / 'ehr_processed.pkl')
        except:
             print("Please run main.py first to generate baseline features.")
             return

        # Mock ECG/MRI features for demonstration if main.py hasn't been run
        n_samples = len(ehr_processed['X_train']) + len(ehr_processed['X_test'])
        ecg_features = {'features': pd.DataFrame(np.random.rand(n_samples, 14), columns=[f'ecg_f{i}' for i in range(14)])}
        mri_features = {'features': pd.DataFrame(np.random.rand(n_samples, 15), columns=[f'mri_f{i}' for i in range(15)])}
        
        fusion = SyntheticFusion(config)
        fused_df = fusion.create_fused_dataset(ehr_processed, ecg_features, mri_features)
    else:
        fused_df = pd.read_csv(fused_path)

    # 2. Extract Modality Features
    clinical_cols = [c for c in fused_df.columns if c.startswith('clinical_')]
    ecg_cols = [c for c in fused_df.columns if c.startswith('ecg_')]
    mri_cols = [c for c in fused_df.columns if c.startswith('mri_')]
    
    X_clinical = fused_df[clinical_cols].values
    X_ecg = fused_df[ecg_cols].values
    X_mri = fused_df[mri_cols].values
    
    # Target 1: Risk Category (0, 1, 2)
    y_risk = fused_df['final_target_encoded'].values
    
    # Target 2: Progression Score (0.0 - 1.0)
    # If not present, derive it from risk_score
    y_progression = fused_df.get('final_risk_score', np.random.rand(len(fused_df))).values
    
    # Target 3: Modality Burdens (0.0 - 1.0)
    # If the synthetic dataset doesn't have explicit severity labels per branch, 
    # we simulate them based on the risk progression or leave as pseudo-targets for prototype
    y_clinical_burden = fused_df.get('clinical_abnormal_count', fused_df['final_target_encoded']).values / 3.0
    y_electrical_burden = np.random.uniform(0.1, 0.9, size=len(fused_df)) 
    y_structural_burden = np.random.uniform(0.1, 0.9, size=len(fused_df))
    # Clip to [0,1]
    y_clinical_burden = np.clip(y_clinical_burden, 0, 1)

    # 3. Scaling
    scaler_clinical = StandardScaler()
    scaler_ecg = StandardScaler()
    scaler_mri = StandardScaler()
    
    X_clinical = scaler_clinical.fit_transform(X_clinical)
    X_ecg = scaler_ecg.fit_transform(X_ecg)
    X_mri = scaler_mri.fit_transform(X_mri)
    
    # Save scalers for API
    joblib.dump({
        'clinical': scaler_clinical,
        'ecg': scaler_ecg,
        'mri': scaler_mri,
        'clinical_cols': clinical_cols,
        'ecg_cols': ecg_cols,
        'mri_cols': mri_cols
    }, config.MODELS_DIR / 'cardiac_fm_scalers.pkl')

    # 4. Split
    indices = np.arange(len(fused_df))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_risk)
    
    # Pack targets into a single tensor array to pass to Dataset
    targets_train = np.column_stack((y_risk[idx_train], y_progression[idx_train], y_clinical_burden[idx_train], y_electrical_burden[idx_train], y_structural_burden[idx_train]))
    targets_test = np.column_stack((y_risk[idx_test], y_progression[idx_test], y_clinical_burden[idx_test], y_electrical_burden[idx_test], y_structural_burden[idx_test]))

    train_dataset = MultimodalDataset(X_clinical[idx_train], X_ecg[idx_train], X_mri[idx_train], targets_train[:,0], targets_train[:,1:])
    test_dataset = MultimodalDataset(X_clinical[idx_test], X_ecg[idx_test], X_mri[idx_test], targets_test[:,0], targets_test[:,1:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 5. Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CardiacFM(
        ehr_dim=len(clinical_cols),
        ecg_dim=len(ecg_cols),
        mri_dim=len(mri_cols)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_risk = nn.CrossEntropyLoss()
    criterion_prog = nn.MSELoss()

    # 6. Training Loop
    epochs = 50
    best_loss = float('inf')
    print(f"\nTraining Cardiac-FM for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for ehr, ecg, mri, risk, aux_targets in train_loader:
            ehr, ecg, mri, risk, aux_targets = ehr.to(device), ecg.to(device), mri.to(device), risk.to(device), aux_targets.to(device)
            # aux_targets is [batch, 4] -> progression, clinical, electrical, structural
            
            optimizer.zero_grad()
            outputs = model(ehr, ecg, mri)
            
            loss_risk = criterion_risk(outputs['risk_logits'], risk)
            loss_prog = criterion_prog(outputs['progression_score'], aux_targets[:, 0].unsqueeze(1))
            loss_clin = criterion_prog(outputs['clinical_score'], aux_targets[:, 1].unsqueeze(1))
            loss_elec = criterion_prog(outputs['electrical_score'], aux_targets[:, 2].unsqueeze(1))
            loss_struc = criterion_prog(outputs['structural_score'], aux_targets[:, 3].unsqueeze(1))
            
            # Combine losses (Risk + Progression + 3x Branch Auxiliary)
            loss = loss_risk + loss_prog + 0.5 * (loss_clin + loss_elec + loss_struc)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for ehr, ecg, mri, risk, aux_targets in test_loader:
                ehr, ecg, mri, risk, aux_targets = ehr.to(device), ecg.to(device), mri.to(device), risk.to(device), aux_targets.to(device)
                outputs = model(ehr, ecg, mri)
                
                l_r = criterion_risk(outputs['risk_logits'], risk)
                l_p = criterion_prog(outputs['progression_score'], aux_targets[:, 0].unsqueeze(1))
                l_c = criterion_prog(outputs['clinical_score'], aux_targets[:, 1].unsqueeze(1))
                l_e = criterion_prog(outputs['electrical_score'], aux_targets[:, 2].unsqueeze(1))
                l_s = criterion_prog(outputs['structural_score'], aux_targets[:, 3].unsqueeze(1))
                
                val_loss += (l_r + l_p + 0.5 * (l_c + l_e + l_s)).item()
                
                _, predicted = torch.max(outputs['risk_logits'].data, 1)
                total += risk.size(0)
                correct += (predicted == risk).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'cardiac_fm.pth')

    print(f"\nTraining complete. Best validation loss: {best_loss:.4f}")
    print(f"Model saved to {config.MODELS_DIR / 'cardiac_fm.pth'}")

if __name__ == "__main__":
    train_model()
