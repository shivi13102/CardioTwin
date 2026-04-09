# src/evaluation/compare_all_models.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .uncertainty_metrics import calculate_ece

class ModelComparator:
    def __init__(self, config):
        self.config = config
        self.results = []

    def evaluate_model(self, name, model, X_test, y_test):
        try:
            # Predictions
            y_prob = model.predict_proba(X_test)
            
            # Binary fallback for 3rd-class models on binary diagnostic tasks (ECG/MRI)
            if y_prob.shape[1] > 2 and len(np.unique(y_test)) == 2:
                y_prob_eval = np.column_stack([y_prob[:, 0], y_prob[:, 1:].sum(axis=1)])
                y_pred = np.argmax(y_prob_eval, axis=1)
                auc = roc_auc_score(y_test, y_prob_eval[:, 1])
            elif y_prob.shape[1] == 2:
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                
            # Basic Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
            # Expected Calibration Error (ECE) for Uncertainty
            ece = calculate_ece(y_test, y_prob)
            
            return {
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': auc,
                'ECE': ece
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            return None

    def evaluate_all(self, models_dict, data_map):
        print("\n=== Evaluating All Models ===\n")
        self.results = []
        for name, model in models_dict.items():
            modality = name.split('-')[0].lower() if '-' in name else 'multimodal'
            if modality in data_map:
                X_t, y_t = data_map[modality]
                res = self.evaluate_model(name, model, X_t, y_t)
                if res:
                    self.results.append(res)
        
        return pd.DataFrame(self.results)

    def plot_comparison(self, results_df):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        os.makedirs(self.config.VISUALIZATIONS_DIR, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='F1-Score', y='Model', data=results_df.sort_values('F1-Score', ascending=False))
        plt.title('Model Performance Comparison (F1-Score)')
        plt.tight_layout()
        plt.savefig(f"{self.config.VISUALIZATIONS_DIR}/all_models_f1_comparison.png")
        plt.close()
