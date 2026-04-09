# src/fusion/clinical_aligner.py
"""
Clinical alignment and synthetic fusion module.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import random
from tqdm import tqdm

class ClinicalAligner:
    """Align modalities based on clinical severity."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def align_modalities(self, ehr_features, ecg_features, mri_features):
        """Align modalities using clinical severity groups."""
        print("\n=== Aligning Modalities ===")
        
        # Extract risk/severity groups
        ehr_risk = self._extract_ehr_risk(ehr_features)
        ecg_severity = self._extract_ecg_severity(ecg_features)
        mri_severity = self._extract_mri_severity(mri_features)
        
        # Create alignment mapping
        alignment_map = self._create_alignment_map(ehr_risk, ecg_severity, mri_severity)
        
        # Save alignment map
        joblib.dump(alignment_map, self.config.FUSED_DATA_DIR / 'alignment_map.pkl')
        
        print("Alignment completed successfully")
        return alignment_map
    
    def _extract_ehr_risk(self, ehr_features):
        """Extract clinical risk groups from EHR."""
        if 'risk_groups' in ehr_features:
            return ehr_features['risk_groups']
        
        # If not available, compute from features
        X_train = ehr_features['X_train']
        y_train = ehr_features['y_train']
        
        risk_scores = []
        for idx in range(len(X_train)):
            score = 0
            
            if 'age' in X_train.columns and X_train['age'].iloc[idx] > 65:
                score += 2
            if 'chol' in X_train.columns and X_train['chol'].iloc[idx] > 240:
                score += 2
            if 'trestbps' in X_train.columns and X_train['trestbps'].iloc[idx] > 140:
                score += 2
            if y_train.iloc[idx] == 1:
                score += 2
            
            risk_scores.append(score)
        
        # Convert to groups
        risk_groups = []
        for score in risk_scores:
            if score <= 2:
                risk_groups.append(0)  # low
            elif score <= 5:
                risk_groups.append(1)  # moderate
            else:
                risk_groups.append(2)  # high
        
        return np.array(risk_groups)
    
    def _extract_ecg_severity(self, ecg_features):
        """Extract ECG severity groups."""
        df = ecg_features['features']
        
        if 'abnormality_group' in df.columns:
            return df['abnormality_group'].values
        
        if 'severity_group' in df.columns:
            severity_map = {'normal': 0, 'mild': 1, 'severe': 2}
            return df['severity_group'].map(severity_map).values
        
        # If not available, compute from features
        scores = df.get('abnormality_score', 0) + df.get('arrhythmia_burden', 0) * 2
        severity_groups = pd.cut(scores, bins=3, labels=[0, 1, 2])
        
        return severity_groups.values
    
    def _extract_mri_severity(self, mri_features):
        """Extract MRI severity groups."""
        df = mri_features['features']
        
        if 'severity_group_encoded' in df.columns:
            return df['severity_group_encoded'].values
        
        if 'severity_group' in df.columns:
            severity_map = {'normal': 0, 'remodeling': 1, 'dysfunction': 2}
            return df['severity_group'].map(severity_map).values
        
        # If not available, compute from features
        scores = df.get('dysfunction_score', 0) + df.get('structural_abnormality', 0)
        severity_groups = pd.cut(scores, bins=3, labels=[0, 1, 2])
        
        return severity_groups.values
    
    def _create_alignment_map(self, ehr_risk, ecg_severity, mri_severity):
        """Create mapping between modality indices based on severity."""
        alignment_map = {
            'low': {'ehr_indices': [], 'ecg_indices': [], 'mri_indices': []},
            'moderate': {'ehr_indices': [], 'ecg_indices': [], 'mri_indices': []},
            'high': {'ehr_indices': [], 'ecg_indices': [], 'mri_indices': []}
        }
        
        # Group indices by severity
        for idx, risk in enumerate(ehr_risk):
            if risk == 0:
                alignment_map['low']['ehr_indices'].append(idx)
            elif risk == 1:
                alignment_map['moderate']['ehr_indices'].append(idx)
            else:
                alignment_map['high']['ehr_indices'].append(idx)
        
        for idx, severity in enumerate(ecg_severity):
            if severity == 0:
                alignment_map['low']['ecg_indices'].append(idx)
            elif severity == 1:
                alignment_map['moderate']['ecg_indices'].append(idx)
            else:
                alignment_map['high']['ecg_indices'].append(idx)
        
        for idx, severity in enumerate(mri_severity):
            if severity == 0:
                alignment_map['low']['mri_indices'].append(idx)
            elif severity == 1:
                alignment_map['moderate']['mri_indices'].append(idx)
            else:
                alignment_map['high']['mri_indices'].append(idx)
        
        # Print alignment statistics
        for group in alignment_map:
            print(f"\n{group.upper()} Risk Group:")
            print(f"  EHR samples: {len(alignment_map[group]['ehr_indices'])}")
            print(f"  ECG samples: {len(alignment_map[group]['ecg_indices'])}")
            print(f"  MRI samples: {len(alignment_map[group]['mri_indices'])}")
        
        return alignment_map

class SyntheticFusion:
    """Create synthetic fused patient profiles."""
    
    def __init__(self, config):
        self.config = config
        self.aligner = ClinicalAligner(config)
        
    def create_fused_dataset(self, ehr_features, ecg_features, mri_features, alignment_map=None):
        """Create synthetic fused dataset."""
        print("\n=== Creating Synthetic Fused Dataset ===")
        
        if alignment_map is None:
            alignment_map = self.aligner.align_modalities(ehr_features, ecg_features, mri_features)
        
        # Extract data
        ehr_data = self._prepare_ehr_data(ehr_features)
        ecg_data = self._prepare_ecg_data(ecg_features)
        mri_data = self._prepare_mri_data(mri_features)
        
        # Create fused profiles
        fused_profiles = []
        
        for group in ['low', 'moderate', 'high']:
            print(f"\nCreating {group} risk profiles...")
            
            ehr_indices = alignment_map[group]['ehr_indices']
            ecg_indices = alignment_map[group]['ecg_indices']
            mri_indices = alignment_map[group]['mri_indices']
            
            # If a modality lacks samples for this risk group, fallback to using all of its samples
            # This is critical for disjoint datasets to ensure we can still form multimodal profiles
            if not ehr_indices: ehr_indices = list(range(len(ehr_data)))
            if not ecg_indices: ecg_indices = list(range(len(ecg_data)))
            if not mri_indices: mri_indices = list(range(len(mri_data)))
            
            # Determine number of profiles to create (use the largest available to not waste data)
            n_profiles = max(len(ehr_indices), len(ecg_indices), len(mri_indices))
            
            # Create profiles by combining compatible samples
            for i in tqdm(range(n_profiles), desc=f"{group} profiles"):
                # Select samples
                ehr_idx = random.choice(ehr_indices)
                ecg_idx = random.choice(ecg_indices)
                mri_idx = random.choice(mri_indices)
                
                # Create fused profile
                profile = self._create_profile(
                    ehr_data.iloc[ehr_idx] if hasattr(ehr_data, 'iloc') else ehr_data[ehr_idx],
                    ecg_data.iloc[ecg_idx] if hasattr(ecg_data, 'iloc') else ecg_data[ecg_idx],
                    mri_data.iloc[mri_idx] if hasattr(mri_data, 'iloc') else mri_data[mri_idx],
                    group
                )
                
                fused_profiles.append(profile)
        
        # Create DataFrame
        fused_df = pd.DataFrame(fused_profiles)
        
        # Create final target label
        fused_df = self._create_target_label(fused_df)
        
        # Save fused dataset
        fused_df.to_csv(self.config.FUSED_DATA_DIR / 'fused_multimodal_dataset.csv', index=False)
        
        # Save metadata
        metadata = {
            'n_profiles': len(fused_df),
            'feature_names': fused_df.columns.tolist(),
            'risk_distribution': fused_df['risk_group'].value_counts().to_dict(),
            'target_distribution': fused_df['final_target'].value_counts().to_dict()
        }
        
        joblib.dump(metadata, self.config.FUSED_DATA_DIR / 'fused_metadata.pkl')
        
        print(f"\n=== Fusion Complete ===")
        print(f"Created {len(fused_df)} synthetic fused profiles")
        print(f"Feature shape: {fused_df.shape}")
        print(f"\nRisk group distribution:")
        print(fused_df['risk_group'].value_counts())
        print(f"\nTarget distribution:")
        print(fused_df['final_target'].value_counts())
        
        return fused_df
    
    def _prepare_ehr_data(self, ehr_features):
        """Prepare EHR data for fusion."""
        # Combine training and test data
        X_train = ehr_features['X_train']
        X_test = ehr_features['X_test']
        y_train = ehr_features['y_train']
        y_test = ehr_features['y_test']
        
        # Combine into single DataFrame
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        
        # Add target
        X_combined['target'] = y_combined
        
        return X_combined
    
    def _prepare_ecg_data(self, ecg_features):
        """Prepare ECG data for fusion."""
        return ecg_features['features']
    
    def _prepare_mri_data(self, mri_features):
        """Prepare MRI data for fusion."""
        return mri_features['features']
    
    def _create_profile(self, ehr_row, ecg_row, mri_row, risk_group):
        """Create a single fused profile dynamically."""
        profile = {}
        
        # Add identifier
        profile['synthetic_patient_id'] = f"{risk_group}_{hash((hash(str(ehr_row)), hash(str(ecg_row)), hash(str(mri_row)))) % 10000:04d}"
        profile['risk_group'] = risk_group
        
        # Add ALL EHR features (clinical)
        for feat in ehr_row.index:
            if feat not in ['target', 'risk_group', 'synthetic_patient_id']:
                prefix = "" if feat.startswith('clinical_') else "clinical_"
                profile[f"{prefix}{feat}"] = ehr_row[feat]
        
        # Add ALL ECG features (electrical)
        for feat in ecg_row.index:
            if feat not in ['abnormality_group', 'severity_group', 'risk_group']:
                prefix = "" if feat.startswith('ecg_') else "ecg_"
                profile[f"{prefix}{feat}"] = ecg_row[feat]
        
        # Add ALL MRI features (structural)
        for feat in mri_row.index:
            if feat not in ['severity_group_encoded', 'severity_group', 'risk_group']:
                prefix = "" if feat.startswith('mri_') else "mri_"
                profile[f"{prefix}{feat}"] = mri_row[feat]
        
        # Add combined features
        profile['fusion_clinical_ecg'] = self._calculate_combined_score(
            profile, [c for c in profile.keys() if 'oldpeak' in c or 'score' in c and 'ecg' in c]
        )
        
        profile['fusion_ecg_mri'] = self._calculate_combined_score(
            profile, [c for c in profile.keys() if ('score' in c or 'ef' in c) and ('ecg' in c or 'mri' in c)]
        )
        
        profile['fusion_severity_index'] = self._calculate_combined_score(
            profile, [c for c in profile.keys() if 'score' in c or 'severity' in c]
        )
        
        return profile
    
    def _calculate_combined_score(self, profile, features):
        """Calculate combined score from multiple features."""
        scores = []
        for feat in features:
            if feat in profile and profile[feat] is not None:
                scores.append(float(profile[feat]))
        
        if scores:
            return np.mean(scores)
        return 0
    
    def _create_target_label(self, df):
        """Create final target label for prediction."""
        if len(df) == 0:
            return df
            
        # Create composite risk score
        risk_score = 0
        
        # Clinical risk
        if 'clinical_oldpeak_severity' in df.columns:
            risk_score += df['clinical_oldpeak_severity'] * 0.3
        
        # ECG abnormality
        if 'ecg_abnormality_score' in df.columns:
            risk_score += df['ecg_abnormality_score'] * 0.3
        
        # MRI dysfunction
        if 'mri_dysfunction_score' in df.columns:
            risk_score += df['mri_dysfunction_score'] * 0.4
        
        # Normalize to 0-1
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-10)
        
        # Create categorical target
        df['final_risk_score'] = risk_score
        df['final_target'] = pd.cut(risk_score, 
                                     bins=[-np.inf, 0.33, 0.67, np.inf], 
                                     labels=['low_risk', 'medium_risk', 'high_risk'])
        
        # Encode target
        target_map = {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}
        df['final_target_encoded'] = df['final_target'].map(target_map)
        
        return df