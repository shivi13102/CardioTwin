import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Setup paths based on existing architecture
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
FUSED_PATH = DATA_DIR / 'fused' / 'fused_multimodal_dataset.csv'
VIZ_DIR = ROOT_DIR / 'frontend' / 'public' / 'viz'

def generate_plots():
    # Ensure visual output directory exists in the React public folder
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    if not FUSED_PATH.exists():
        print(f"Error: {FUSED_PATH} not found. Cannot generate EDA.")
        print("Please ensure the data pipeline is executed to produce fused data.")
        return

    print("Loading fused multimodal dataset...")
    df = pd.read_csv(FUSED_PATH)

    clinical_cols = [c for c in df.columns if c.startswith('clinical_')]
    ecg_cols = [c for c in df.columns if c.startswith('ecg_')]
    mri_cols = [c for c in df.columns if c.startswith('mri_')]

    # 1. Target Class Distribution
    print("Generating Risk Category Distribution...")
    plt.figure(figsize=(7, 5))
    if 'final_target_encoded' in df.columns:
        ax = sns.countplot(data=df, x='final_target_encoded', palette='viridis')
        plt.title('Risk Category Distribution in Dataset', pad=15)
        plt.xlabel('Risk Category (0 = Low, 1 = Moderate, 2 = High)')
        plt.ylabel('Patient Count')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'risk_distribution.png', dpi=200)
    plt.close()

    # 2. EHR Correlation Heatmap
    print("Generating EHR/Clinical Correlation Matrix...")
    if clinical_cols:
        plt.figure(figsize=(10, 8))
        # Take the top 15 features to avoid clutter
        corr = df[clinical_cols[:15]].corr()
        # Clean labels
        corr.columns = [c.replace('clinical_', '') for c in corr.columns]
        corr.index = [i.replace('clinical_', '') for i in corr.index]
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt=".1f", square=True, annot_kws={'size': 8})
        plt.title('EHR (Clinical) Feature Correlation', pad=15)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'ehr_corr.png', dpi=200)
    plt.close()

    # 3. ECG Correlation Heatmap
    print("Generating ECG Correlation Matrix...")
    if ecg_cols:
        plt.figure(figsize=(10, 8))
        corr = df[ecg_cols].corr()
        corr.columns = [c.replace('ecg_', '') for c in corr.columns]
        corr.index = [i.replace('ecg_', '') for i in corr.index]
        sns.heatmap(corr, cmap='YlOrRd', center=0, annot=True, fmt=".1f", square=True, annot_kws={'size': 8})
        plt.title('ECG Feature Correlation', pad=15)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'ecg_corr.png', dpi=200)
    plt.close()

    # 4. MRI Correlation Heatmap
    print("Generating MRI Correlation Matrix...")
    if mri_cols:
        plt.figure(figsize=(10, 8))
        corr = df[mri_cols[:15]].corr()
        corr.columns = [c.replace('mri_', '') for c in corr.columns]
        corr.index = [i.replace('mri_', '') for i in corr.index]
        sns.heatmap(corr, cmap='mako', center=0, annot=True, fmt=".1f", square=True, annot_kws={'size': 8})
        plt.title('Cardiac MRI Feature Correlation', pad=15)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'mri_corr.png', dpi=200)
    plt.close()

    # 5. Multimodal Cross-Correlation
    print("Generating Multimodal Cross-Correlation Matrix...")
    cross_cols = clinical_cols[:5] + ecg_cols[:5] + mri_cols[:5]
    if cross_cols and 'final_target_encoded' in df.columns:
        cross_cols.append('final_target_encoded')
        plt.figure(figsize=(12, 10))
        corr = df[cross_cols].corr()
        # Clean labels for cross-modality
        clean_cols = [c.replace('clinical_', 'EHR_').replace('ecg_', 'ECG_').replace('mri_', 'MRI_').replace('final_target_encoded', 'TARGET_RISK') for c in corr.columns]
        corr.columns = clean_cols
        corr.index = clean_cols
        sns.heatmap(corr, cmap='Spectral', center=0, cbar=True, annot=True, fmt=".1f", annot_kws={'size': 7})
        plt.title('Multimodal Cross-Feature Correlation (EHR vs ECG vs MRI vs Target)', pad=15)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'multimodal_corr.png', dpi=200)
    plt.close()

    print(f"✅ Generated plots successfully in {VIZ_DIR}")

if __name__ == "__main__":
    generate_plots()
