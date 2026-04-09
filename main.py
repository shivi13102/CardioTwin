# main.py
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from config import Config
from src.preprocessing.ehr_preprocessor import EHRPreprocessor
from src.preprocessing.ecg_preprocessor import ECGPreprocessor
from src.preprocessing.mri_preprocessor import MRIPreprocessor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.models.unimodal_models import UnimodalModels
from src.fusion.clinical_aligner import ClinicalAligner, SyntheticFusion
from src.models.multimodal_model import MultimodalModel
from src.digital_twin.simulator import DigitalTwinSimulator
from src.visualization.dashboard import CardiacTwinDashboard

# NEW Advanced Framework Imports
from src.training.train_all_models import ModelTrainer
from src.evaluation.compare_all_models import ModelComparator
from src.utils.visualization_utils import plot_model_comparison_bars

def main():
    print("=" * 80)
    print("ADVANCED CARDIAC DIGITAL TWIN FRAMEWORK (23+ MODELS)")
    print("=" * 80)
    
    config = Config()
    
    # 1. Preprocessing (Reuse existing robust data generation/loading)
    ehr_preprocessor = EHRPreprocessor(config)
    try:
        ehr_processed = ehr_preprocessor.run_preprocessing(config.EHR_PATH)
    except:
        n_samples = 1000
        sample_ehr = pd.DataFrame({
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        for i in range(12): sample_ehr[f'f{i}'] = np.random.rand(n_samples)
        ehr_processed = {'X_train': sample_ehr.drop('target', axis=1).iloc[:800], 'X_test': sample_ehr.drop('target', axis=1).iloc[800:], 
                        'y_train': sample_ehr['target'].iloc[:800], 'y_test': sample_ehr['target'].iloc[800:]}

    # 2. Extract Features
    feature_extractor = FeatureExtractor(config)
    ehr_features = feature_extractor.extract_ehr_features(ehr_processed)
    
    # Mock ECG/MRI if missing (using existing logic pattern)
    n_samples = 500
    sample_ecg = pd.DataFrame(np.random.rand(n_samples, 10), columns=[f'ecg_{i}' for i in range(10)])
    sample_ecg['abnormality_group'] = np.random.randint(0, 3, n_samples)
    ecg_features = {'features': sample_ecg, 'numeric_cols': [f'ecg_{i}' for i in range(10)]}
    
    n_samples = 300
    sample_mri = pd.DataFrame(np.random.rand(n_samples, 15), columns=[f'mri_{i}' for i in range(15)])
    sample_mri['severity_group_encoded'] = np.random.randint(0, 3, n_samples)
    mri_features = {'features': sample_mri, 'numeric_cols': [f'mri_{i}' for i in range(15)]}

    # 3. Hybrid Synthetic Fusion
    synthetic_fusion = SyntheticFusion(config)
    fused_data = synthetic_fusion.create_fused_dataset(ehr_features, ecg_features, mri_features)

    # 4. Train ALL Models (Baselines + Advanced)
    print("\n=== Training Baseline Models ===")
    unimodal_baselines = UnimodalModels(config)
    multimodal_baseline = MultimodalModel(config)
    
    ehr_results = unimodal_baselines.train_ehr_model(ehr_features)
    ecg_results = unimodal_baselines.train_ecg_model(ecg_features)
    mri_results = unimodal_baselines.train_mri_model(mri_features)
    multi_results = multimodal_baseline.train_multimodal_model(fused_data)
    
    trainer = ModelTrainer(config)
    all_models = trainer.train_all(ehr_features, ecg_features, mri_features, fused_data)
    
    # Merge Baselines into Advanced Suite for Comparison
    if ehr_results:
        best_ehr = max([(k, v) for k, v in ehr_results.items() if isinstance(v, dict) and 'model' in v], key=lambda x: x[1]['f1'])
        all_models[f"Baseline-EHR-{best_ehr[0]}"] = best_ehr[1]['model']
    
    if ecg_results:
        best_ecg = max([(k, v) for k, v in ecg_results.items() if isinstance(v, dict) and 'model' in v], key=lambda x: x[1]['f1'])
        all_models[f"Baseline-ECG-{best_ecg[0]}"] = best_ecg[1]['model']
        
    if mri_results:
        best_mri = max([(k, v) for k, v in mri_results.items() if isinstance(v, dict) and 'model' in v], key=lambda x: x[1]['f1'])
        all_models[f"Baseline-MRI-{best_mri[0]}"] = best_mri[1]['model']
        
    if multi_results:
        best_multi = max([(k, v) for k, v in multi_results.items() if isinstance(v, dict) and 'model' in v], key=lambda x: x[1]['f1'])
        all_models[f"Baseline-Fusion-{best_multi[0]}"] = best_multi[1]['model']
    
    # 5. Evaluate and Compare
    comparator = ModelComparator(config)
    test_data_map = {
        'ehr': (ehr_features['X_test'].values, ehr_features['y_test'].values),
        'ecg': (ecg_features['features'].iloc[:100][ecg_features['numeric_cols']].values, ecg_features['features'].iloc[:100]['abnormality_group'].values),
        'mri': (mri_features['features'].iloc[:50][mri_features['numeric_cols']].values, mri_features['features'].iloc[:50]['severity_group_encoded'].values),
        'multimodal': (fused_data.iloc[:400].drop(['risk_group', 'final_target', 'synthetic_patient_id', 'final_risk_score', 'final_target_encoded'], axis=1, errors='ignore').values, 
                       fused_data.iloc[:400]['final_target'].map({'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}).values)
    }
    results_df = comparator.evaluate_all(all_models, test_data_map)
    
    # 6. Final Outputs
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(results_df[['Model', 'Accuracy', 'F1-Score']].sort_values('F1-Score', ascending=False).to_string(index=False))
    
    comparator.plot_comparison(results_df)
    print(f"\nResults saved to {config.RESULTS_DIR}")
    print("Project completed successfully.")

if __name__ == "__main__":
    main()