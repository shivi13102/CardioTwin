# src/preprocessing/ehr_preprocessor.py
"""
EHR data preprocessing module.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EHRPreprocessor:
    """Preprocessor for EHR (heart disease) dataset."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load EHR dataset and handle column mapping."""
        print(f"Loading EHR data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        
        # Drop identifier if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        # Rename target column (UCi dataset uses 'num')
        if 'num' in df.columns:
            df = df.rename(columns={'num': 'target'})
            print("Renamed 'num' to 'target'")
            
        # Handle thalch (UCi) vs thalach (code) mapping
        if 'thalch' in df.columns:
            df = df.rename(columns={'thalch': 'thalach'})
            print("Renamed 'thalch' to 'thalach'")
            
        # Binarize target (0: normal, 1-4: disease)
        if 'target' in df.columns:
            df['target'] = (df['target'] > 0).astype(int)
            print("Binarized target variable (0: normal, 1+: disease)")
            
        print(f"Final columns: {df.columns.tolist()}")
        return df
    
    def inspect_data(self, df):
        """Inspect dataset for missing values and data types."""
        print("\n=== Data Inspection ===")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nTarget distribution:\n{df['target'].value_counts()}")
        
    def clean_data(self, df):
        """Clean the dataset."""
        print("\n=== Cleaning Data ===")
        
        # Handle missing values
        # For numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Impute numerical missing values with median
        if numerical_cols:
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        # Impute categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        
        print(f"Missing values after imputation:\n{df.isnull().sum()}")
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables."""
        print("\n=== Encoding Categorical Variables ===")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col != 'target':  # Skip target if categorical
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        return df
    
    def create_engineered_features(self, df):
        """Create engineered features."""
        print("\n=== Creating Engineered Features ===")
        
        # Age group categorization
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 40, 55, 65, 100], 
                                     labels=['young', 'middle', 'senior', 'elderly'])
            df['age_group'] = df['age_group'].cat.codes
        
        # Cholesterol risk level
        if 'chol' in df.columns:
            df['cholesterol_risk'] = pd.cut(df['chol'],
                                           bins=[0, 200, 240, 600],
                                           labels=['normal', 'borderline', 'high'])
            df['cholesterol_risk'] = df['cholesterol_risk'].cat.codes
        
        # Blood pressure category
        if 'trestbps' in df.columns:
            df['bp_category'] = pd.cut(df['trestbps'],
                                       bins=[0, 120, 130, 140, 200],
                                       labels=['normal', 'elevated', 'high_stage1', 'high_stage2'])
            df['bp_category'] = df['bp_category'].cat.codes
        
        # Age × Cholesterol interaction
        if 'age' in df.columns and 'chol' in df.columns:
            df['age_chol_interaction'] = df['age'] * df['chol'] / 1000
        
        # Max heart rate ratio
        if 'thalach' in df.columns and 'age' in df.columns:
            df['max_hr_ratio'] = df['thalach'] / (220 - df['age'])
        
        # Oldpeak severity
        if 'oldpeak' in df.columns:
            df['oldpeak_severity'] = pd.cut(df['oldpeak'],
                                           bins=[-1, 0.5, 1.5, 10],
                                           labels=['mild', 'moderate', 'severe'])
            df['oldpeak_severity'] = df['oldpeak_severity'].cat.codes
        
        # Number of abnormal indicators
        abnormal_cols = ['exang', 'fbs', 'oldpeak_severity']
        abnormal_available = [col for col in abnormal_cols if col in df.columns]
        if abnormal_available:
            df['abnormal_count'] = df[abnormal_available].sum(axis=1)
        
        print(f"New features created: {[col for col in df.columns if col not in self.feature_names] if self.feature_names else 'All features'}")
        return df
    
    def normalize_features(self, df, feature_cols):
        """Normalize numerical features."""
        print("\n=== Normalizing Features ===")
        
        # Select numerical columns that should be normalized
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        print(f"Normalized {len(numerical_cols)} numerical features")
        return df_scaled
    
    def prepare_features(self, df, target_col='target'):
        """Prepare features for modeling."""
        print("\n=== Preparing Features ===")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle class imbalance
        from collections import Counter
        print(f"Class distribution: {Counter(y)}")
        
        # Normalize features
        X_normalized = self.normalize_features(X, X.columns)
        
        return X_normalized, y
    
    def run_preprocessing(self, filepath):
        """Run complete preprocessing pipeline."""
        # Load data
        df = self.load_data(filepath)
        
        # Inspect data
        self.inspect_data(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        # Create engineered features
        df = self.create_engineered_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        # Save processed data
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        # Save to file
        import joblib
        joblib.dump(processed_data, self.config.PROCESSED_DATA_DIR / 'ehr_processed.pkl')
        
        print(f"\n=== Preprocessing Complete ===")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return processed_data