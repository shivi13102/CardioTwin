# src/feature_extraction/feature_extractor.py
"""
Unified feature extraction for all modalities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class FeatureExtractor:
    """Unified feature extraction for multimodal fusion."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_ehr_features(self, ehr_data):
        """Extract and engineer EHR features."""
        print("\n=== Extracting EHR Features ===")
        
        # Load processed EHR data
        if isinstance(ehr_data, str):
            ehr_processed = joblib.load(ehr_data)
            X_train = ehr_processed['X_train']
            X_test = ehr_processed['X_test']
            y_train = ehr_processed['y_train']
            y_test = ehr_processed['y_test']
            feature_names = ehr_processed['feature_names']
        else:
            # Assume ehr_data is processed data dict
            X_train = ehr_data['X_train']
            X_test = ehr_data['X_test']
            y_train = ehr_data['y_train']
            y_test = ehr_data['y_test']
            feature_names = ehr_data['feature_names']
        
        # Create clinical risk groups for train and test
        train_risk_groups_encoded = self._create_clinical_risk_groups(X_train, y_train)
        test_risk_groups_encoded = self._assign_risk_groups(X_test, y_test)
        
        # Combine features with risk groups
        X_train_with_risk = X_train.copy()
        X_train_with_risk['clinical_risk_group'] = train_risk_groups_encoded
        
        X_test_with_risk = X_test.copy()
        X_test_with_risk['clinical_risk_group'] = test_risk_groups_encoded
        
        # Prepare final feature set
        ehr_features = {
            'X_train': X_train_with_risk,
            'X_test': X_test_with_risk,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names + ['clinical_risk_group'],
            'risk_groups': np.concatenate([train_risk_groups_encoded, test_risk_groups_encoded])
        }
        
        # Save features
        joblib.dump(ehr_features, self.config.PROCESSED_DATA_DIR / 'ehr_features.pkl')
        
        print(f"EHR features shape: {X_train_with_risk.shape}")
        return ehr_features
    
    def _calculate_risk_scores(self, X, y=None):
        """Calculate clinical risk scores based on features."""
        risk_scores = []
        for idx in range(len(X)):
            score = 0
            # Age risk
            if 'age' in X.columns and X['age'].iloc[idx] > 65:
                score += 2
            elif 'age' in X.columns and X['age'].iloc[idx] > 50:
                score += 1
            # Cholesterol risk
            if 'chol' in X.columns and X['chol'].iloc[idx] > 240:
                score += 2
            elif 'chol' in X.columns and X['chol'].iloc[idx] > 200:
                score += 1
            # Blood pressure risk
            if 'trestbps' in X.columns and X['trestbps'].iloc[idx] > 140:
                score += 2
            elif 'trestbps' in X.columns and X['trestbps'].iloc[idx] > 120:
                score += 1
            # Oldpeak risk
            if 'oldpeak' in X.columns and X['oldpeak'].iloc[idx] > 1.5:
                score += 2
            elif 'oldpeak' in X.columns and X['oldpeak'].iloc[idx] > 0.5:
                score += 1
            # Target label (if available)
            if y is not None and y.iloc[idx] == 1:
                score += 2
            risk_scores.append(score)
        return risk_scores

    def _convert_scores_to_groups(self, risk_scores):
        """Convert risk scores to categorical groups."""
        risk_groups = []
        for score in risk_scores:
            if score <= 2:
                risk_groups.append('low')
            elif score <= 5:
                risk_groups.append('moderate')
            else:
                risk_groups.append('high')
        return risk_groups

    def _create_clinical_risk_groups(self, X, y):
        """Create clinical risk groups based on features and labels."""
        risk_scores = self._calculate_risk_scores(X, y)
        risk_groups = self._convert_scores_to_groups(risk_scores)
        
        # Encode risk groups
        le = LabelEncoder()
        risk_groups_encoded = le.fit_transform(risk_groups)
        self.label_encoders['clinical_risk'] = le
        return risk_groups_encoded
    
    def _assign_risk_groups(self, X, y=None):
        """Assign risk groups to data deterministically."""
        risk_scores = self._calculate_risk_scores(X, y)
        risk_groups = self._convert_scores_to_groups(risk_scores)
        
        # Use existing encoder if available
        if 'clinical_risk' in self.label_encoders:
            return self.label_encoders['clinical_risk'].transform(risk_groups)
        else:
            le = LabelEncoder()
            encoded = le.fit_transform(risk_groups)
            self.label_encoders['clinical_risk'] = le
            return encoded
    
    def extract_ecg_features(self, ecg_data):
        """Extract and engineer ECG features."""
        print("\n=== Extracting ECG Features ===")
        
        # Load processed ECG data
        if isinstance(ecg_data, str):
            ecg_processed = joblib.load(ecg_data)
            df = ecg_processed['features']
        else:
            df = ecg_data['features']
        
        # Debug info
        print(f"ECG DataFrame columns: {df.columns.tolist()}")
        if df.empty:
            print("WARNING: ECG DataFrame is empty!")
            return None

        # Create abnormality groups
        if 'severity_group' not in df.columns:
            print("WARNING: 'severity_group' missing from ECG data. Defaulting to 'normal'.")
            df['severity_group'] = 'normal'

        df['abnormality_group'] = df['severity_group'].map({
            'normal': 0,
            'mild': 1,
            'severe': 2
        })
        
        # Create rhythm stability score
        df['rhythm_stability_score'] = 1 / (df.get('rr_cv', 1) + 0.1)
        
        # Create combined arrhythmia burden
        df['arrhythmia_burden'] = (
            df.get('ectopic_percentage', 0) / 100 +
            df.get('irregularity_score', 0) +
            df.get('abnormality_score', 0) / 3
        ) / 3
        
        # Normalize features - ensure we don't leak the targets into the feature set!
        leakage_cols = ['record_id', 'severity_group', 'abnormality_group', 'abnormality_score', 'arrhythmia_burden']
        feature_cols = [col for col in df.columns if col not in leakage_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Prepare for fusion
        ecg_features = {
            'features': df_scaled,
            'feature_names': feature_cols,
            'numeric_cols': numeric_cols
        }
        
        # Save features
        joblib.dump(ecg_features, self.config.PROCESSED_DATA_DIR / 'ecg_features.pkl')
        
        print(f"ECG features shape: {df_scaled.shape}")
        return ecg_features
    
    def extract_mri_features(self, mri_data):
        """Extract and engineer MRI features."""
        print("\n=== Extracting MRI Features ===")
        
        # Load processed MRI data
        if isinstance(mri_data, str):
            mri_processed = joblib.load(mri_data)
            df = mri_processed['features']
        else:
            df = mri_data['features']
        
        # Debug info
        print(f"MRI DataFrame columns: {df.columns.tolist()}")
        if df.empty:
            print("WARNING: MRI DataFrame is empty!")
            return None

        # Create severity groups
        if 'severity_group' not in df.columns:
            print("WARNING: 'severity_group' missing from MRI data. Defaulting to 'normal'.")
            df['severity_group'] = 'normal'

        df['severity_group_encoded'] = df['severity_group'].map({
            'normal': 0,
            'remodeling': 1,
            'dysfunction': 2
        })
        
        # Create combined dysfunction score
        df['dysfunction_score'] = (
            (100 - df.get('lv_ejection_fraction', 55)) / 50 +
            (100 - df.get('rv_ejection_fraction', 55)) / 50 +
            df.get('lv_wall_motion_score', 0) * 2
        ) / 3
        
        # Create structural abnormality index
        df['structural_abnormality'] = (
            (df.get('lv_area', 1000) - 1000) / 1000 +
            (df.get('heart_eccentricity', 0.5) - 0.5) * 2
        ) / 2
        
        # Normalize features - ensure we don't leak targets
        leakage_cols = ['case_id', 'severity_group', 'severity_group_encoded', 'severity_score', 'dysfunction_score', 'structural_abnormality']
        feature_cols = [col for col in df.columns if col not in leakage_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Prepare for fusion
        mri_features = {
            'features': df_scaled,
            'feature_names': feature_cols,
            'numeric_cols': numeric_cols
        }
        
        # Save features
        joblib.dump(mri_features, self.config.PROCESSED_DATA_DIR / 'mri_features.pkl')
        
        print(f"MRI features shape: {df_scaled.shape}")
        return mri_features