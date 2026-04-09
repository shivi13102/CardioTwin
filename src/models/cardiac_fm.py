import torch
import torch.nn as nn
import torch.nn.functional as F

class CardiacFM(nn.Module):
    """
    Cardiac-FM: A Multimodal Foundation Model for Cardiac Digital Twins.
    Integrates EHR, ECG, and MRI features into a shared latent representation.
    """
    def __init__(self, ehr_dim, ecg_dim, mri_dim, latent_dim=64, hidden_dim=128):
        super(CardiacFM, self).__init__()
        
        # 1. EHR Encoder (Clinical)
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 2. ECG Encoder (Electrical)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 3. MRI Encoder (Structural)
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 4. Fusion Module
        # Concatenate 3 * latent_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Gating Layer (Simple Attention)
        self.gate = nn.Linear(hidden_dim, 3) # Weights for each modality
        
        # 5. Prediction Heads
        # Risk classification: Low / Moderate / High
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3) 
        )
        
        # Progression score: 0.0 to 1.0
        self.progression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Modality Burden Heads (Bounded 0.0 to 1.0)
        self.clinical_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.electrical_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.structural_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, ehr, ecg, mri):
        # Encode each modality separately
        ehr_emb = self.ehr_encoder(ehr)
        ecg_emb = self.ecg_encoder(ecg)
        mri_emb = self.mri_encoder(mri)
        
        # Fusion
        fused_raw = torch.cat([ehr_emb, ecg_emb, mri_emb], dim=1)
        twin_state = self.fusion_mlp(fused_raw)
        
        # Optional: Modality weighting/gating for interpretability
        gate_weights = F.softmax(self.gate(twin_state), dim=1)
        
        # Predictions
        risk_logits = self.risk_head(twin_state)
        progression_score = self.progression_head(twin_state)
        
        # Specific Modality Burden Scores
        clinical_score = self.clinical_head(ehr_emb)
        electrical_score = self.electrical_head(ecg_emb)
        structural_score = self.structural_head(mri_emb)
        
        return {
            'risk_logits': risk_logits,
            'progression_score': progression_score,
            'clinical_score': clinical_score,
            'electrical_score': electrical_score,
            'structural_score': structural_score,
            'twin_state': twin_state,
            'modalities': {
                'clinical': ehr_emb,
                'electrical': ecg_emb,
                'structural': mri_emb
            },
            'attention_weights': gate_weights
        }

    def get_twin_state(self, ehr, ecg, mri):
        """Extract the shared latent representation."""
        with torch.no_grad():
            outputs = self.forward(ehr, ecg, mri)
        return outputs['twin_state']
