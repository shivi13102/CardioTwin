# src/training/train_all_models.py
import torch
import numpy as np
import pandas as pd
from src.models import (
    EHR_TFT, EHR_BayesNN, EHR_LSTM_Attention, EHR_GBT_Time,
    ECG_PINN, ECG_Hybrid, ECG_InverseSolver, ECG_OSACN_Net, ECG_CNN_LSTM,
    MRI_MADRU_Net, MRI_SequenceMorph, MRI_VelocityGAN, MRI_ScarMapper, MRI_Diffusion,
    Cardiac_CCAT, Cardiac_GNN, Cardiac_PC_PINN, Cardiac_MCLF, Cardiac_TMF_Net, 
    Cardiac_LDMG, Cardiac_AutoML, Cardiac_BMF_UQ,
    UnimodalModels, MultimodalModel
)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.trained_models = {}

    def train_all(self, ehr_data, ecg_data, mri_data, combined_data):
        print("\n=== Training All Models ===\n")
        
        # EHR
        X_ehr = ehr_data['X_train'].values if hasattr(ehr_data['X_train'], 'values') else ehr_data['X_train']
        y_ehr = ehr_data['y_train'].values if hasattr(ehr_data['y_train'], 'values') else ehr_data['y_train']
        ehr_dim = X_ehr.shape[1]
        
        ehr_models = {
            'EHR-TFT': EHR_TFT(input_dim=ehr_dim),
            'EHR-BayesNN': EHR_BayesNN(input_dim=ehr_dim),
            'EHR-LSTM-Attention': EHR_LSTM_Attention(input_dim=ehr_dim),
            'EHR-GBT-Time': EHR_GBT_Time()
        }
        for name, model in ehr_models.items():
            print(f"- Training {name}..."); model.fit(X_ehr, y_ehr); self.trained_models[name] = model

        # ECG
        X_ecg, y_ecg = ecg_data['features'][ecg_data['numeric_cols']].values, ecg_data['features']['abnormality_group'].values
        ecg_dim = X_ecg.shape[1]
        
        ecg_models = {
            'ECG-PINN': ECG_PINN(input_dim=ecg_dim),
            'ECG-Hybrid': ECG_Hybrid(input_dim=ecg_dim),
            'ECG-InverseSolver': ECG_InverseSolver(input_dim=ecg_dim),
            'ECG-OSACN': ECG_OSACN_Net(input_dim=ecg_dim),
            'ECG-CNN-LSTM': ECG_CNN_LSTM(input_dim=ecg_dim)
        }
        for name, model in ecg_models.items():
            print(f"- Training {name}..."); model.fit(X_ecg, y_ecg); self.trained_models[name] = model

        # MRI
        X_mri, y_mri = mri_data['features'][mri_data['numeric_cols']].values, mri_data['features']['severity_group_encoded'].values
        mri_dim = X_mri.shape[1]
        
        mri_models = {
            'MRI-MADRU-Net': MRI_MADRU_Net(input_dim=mri_dim),
            'MRI-SequenceMorph': MRI_SequenceMorph(input_dim=mri_dim),
            'MRI-VelocityGAN': MRI_VelocityGAN(input_dim=mri_dim),
            'MRI-ScarMapper': MRI_ScarMapper(input_dim=mri_dim),
            'MRI-Diffusion': MRI_Diffusion(input_dim=mri_dim)
        }
        for name, model in mri_models.items():
            print(f"- Training {name}..."); model.fit(X_mri, y_mri); self.trained_models[name] = model

        # Multimodal
        X_multi_df = combined_data.drop(['risk_group', 'final_target', 'synthetic_patient_id', 'final_risk_score', 'final_target_encoded'], axis=1, errors='ignore')
        X_multi = X_multi_df.values
        y_multi = combined_data['final_target'].map({'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}).values
        
        # Detect dimensions for fusion models
        m_cols = X_multi_df.columns
        ehr_dim_m = len([c for c in m_cols if c.startswith('clinical_')])
        ecg_dim_m = len([c for c in m_cols if c.startswith('ecg_')])
        # Any remaining columns (mri_ and fusion_) go into the mri branch for CCAT/MCLF
        mri_dim_m = len(m_cols) - ehr_dim_m - ecg_dim_m
        multi_dim = X_multi.shape[1]
        
        multi_models = {
            'CCAT': Cardiac_CCAT(ehr_dim=ehr_dim_m, ecg_dim=ecg_dim_m, mri_dim=mri_dim_m),
            'Cardiac-GNN': Cardiac_GNN(), # Node features fixed at 2
            'PC-PINN': Cardiac_PC_PINN(input_dim=multi_dim),
            'Contrastive-MCLF': Cardiac_MCLF(ehr_dim=ehr_dim_m, ecg_dim=ecg_dim_m, mri_dim=mri_dim_m),
            'TMF-Net': Cardiac_TMF_Net(input_dim=multi_dim),
            'LDMG': Cardiac_LDMG(input_dim=multi_dim),
            'AutoML': Cardiac_AutoML(input_dim=multi_dim),
            'BMF-UQ': Cardiac_BMF_UQ(input_dim=multi_dim)
        }
        for name, model in multi_models.items():
            print(f"- Training {name}..."); model.fit(X_multi, y_multi); self.trained_models[name] = model

        return self.trained_models

    def save_all(self, directory):
        import os
        os.makedirs(directory, exist_ok=True)
        for name, model in self.trained_models.items():
            model.save(f"{directory}/{name.lower().replace('-', '_')}.pkl")
