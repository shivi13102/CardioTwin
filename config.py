# config.py
"""
Configuration file for the Cardiac Digital Twin project.
"""

import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FUSED_DATA_DIR = DATA_DIR / "fused"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
    
    # Create directories if they don't exist
    for dir_path in [PROCESSED_DATA_DIR, FUSED_DATA_DIR, MODELS_DIR, 
                     RESULTS_DIR, VISUALIZATIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    EHR_PATH = RAW_DATA_DIR / "heart_disease_uci.csv"
    ECG_PATH = RAW_DATA_DIR / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
    MRI_PATH = RAW_DATA_DIR / "ACDC"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # ECG parameters
    ECG_SAMPLING_RATE = 360  # Hz for MIT-BIH
    ECG_SEGMENT_LENGTH = 360  # 1 second segments
    ECG_R_PEAK_WINDOW = 150  # ms before and after R peak
    
    # MRI parameters
    MRI_TARGET_SIZE = (256, 256)
    MRI_NORMALIZATION = "zscore"
    
    # Fusion parameters
    CLINICAL_RISK_GROUPS = ["low", "moderate", "high"]
    ECG_ABNORMALITY_GROUPS = ["normal", "mild", "severe"]
    MRI_SEVERITY_GROUPS = ["normal", "remodeling", "dysfunction"]
    
    # Targets
    TARGET_COLUMN = "target"  # For EHR
    RISK_LEVELS = ["low_risk", "medium_risk", "high_risk"]
    
    # Model hyperparameters
    XGBOOST_PARAMS = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE
    }
    
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE
    }

config = Config()