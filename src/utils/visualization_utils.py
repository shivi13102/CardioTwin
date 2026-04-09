# src/utils/visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_model_comparison_bars(results_df, save_path):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
    plt.title('Advanced Framework: Model Accuracy Comparison')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices_grid(models_results, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)
    for model_name, data in models_results.items():
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_cm.png")
        plt.close()
