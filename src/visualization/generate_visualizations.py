import os
import sys
import json
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
EHR_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'heart_disease_uci.csv')
ECG_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ecg_features.csv')
MRI_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'mri_features.csv')

# Put visualizations precisely here so React can serve them trivially via /visualization/
OUT_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'public', 'visualization')

# Ensure directories exist
for sub in ['ehr', 'ecg', 'mri', 'multimodal', 'summaries']:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

manifest = []

def register_plot(modality, title, path, description):
    """Registers a plot to the manifest."""
    # Convert absolute path to a relative URL-friendly path from public
    rel_path = path.replace(os.path.join(PROJECT_ROOT, 'frontend', 'public', ''), '').replace('\\', '/')
    manifest.append({
        "modality": modality,
        "title": title,
        "path": "/" + rel_path,
        "description": description
    })
    logging.info(f"Saved: {path}")

def clean_dataframe(df, target_col=None, num_cols=None):
    """Safely clean dataframe, dropping NaNs and handling types."""
    if df is None or df.empty: return None
    df = df.copy()
    if 'id' in df.columns: 
        df = df.drop(columns=['id'])
    
    # Coerce to numeric
    for col in df.columns:
        if col != target_col and (num_cols is None or col in num_cols):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaNs in our needed columns
    needed = [target_col] if target_col else []
    if num_cols: needed.extend(num_cols)
    needed = [c for c in needed if c in df.columns]
    
    df = df.dropna(subset=needed)
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    
    return df

# ---------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------
def plot_class_distribution(df, label_col, title, out_path, desc, modality):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=label_col, palette='viridis')
    plt.title(title, pad=15)
    plt.xlabel('Disease / Severity Class')
    plt.ylabel('Count')
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    register_plot(modality, title, out_path, desc)

def plot_histogram_kde(df, features, title, out_path, desc, modality):
    n = len(features)
    if n == 0: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, feat in enumerate(features[:4]):
        if feat in df.columns:
            sns.histplot(df[feat], kde=True, ax=axes[i], color='teal', bins=20)
            axes[i].set_title(feat.replace('_', ' ').title())
            axes[i].set_xlabel('')
    for j in range(i+1, 4):
        fig.delaxes(axes[j])
    plt.suptitle(title, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    register_plot(modality, title, out_path, desc)

def plot_violin(df, features, label_col, title, out_path, desc, modality):
    n = len(features)
    if n == 0: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, feat in enumerate(features[:4]):
        if feat in df.columns and label_col in df.columns:
            sns.violinplot(data=df, x=label_col, y=feat, ax=axes[i], palette='mako', inner='quartile')
            axes[i].set_title(f"{feat.title()} by Class")
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
    for j in range(i+1, 4):
        fig.delaxes(axes[j])
    plt.suptitle(title, y=1.02)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    register_plot(modality, title, out_path, desc)

def plot_scatter(df, x_col, y_col, hue_col, title, out_path, desc, modality):
    if x_col not in df.columns or y_col not in df.columns: return
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='flare', alpha=0.7, edgecolor=None)
    plt.title(title, pad=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    register_plot(modality, title, out_path, desc)

def plot_pca_projection(df, features, label_col, title, out_path, desc, modality):
    valid_feats = [f for f in features if f in df.columns]
    if len(valid_feats) < 2 or label_col not in df.columns: return
    
    X = df[valid_feats].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Label': df[label_col]})
    
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', palette='crest', alpha=0.8, edgecolor='w', s=60)
    plt.title(title, pad=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    register_plot(modality, title, out_path, desc)

# ---------------------------------------------------------
# Generation Pipelines
# ---------------------------------------------------------
def generate_ehr():
    logging.info("Generating EHR visualizations...")
    df = pd.read_csv(EHR_PATH) if os.path.exists(EHR_PATH) else None
    features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df = clean_dataframe(df, target_col='num', num_cols=features)
    if df is None: return None
    
    d_out = os.path.join(OUT_DIR, 'ehr')
    
    plot_class_distribution(df, 'num', "EHR Disease Class Distribution", os.path.join(d_out, 'ehr_class_distribution.png'), "Shows how samples are distributed across severity categories.", "EHR")
    plot_histogram_kde(df, ['age', 'chol', 'trestbps', 'oldpeak'], "Distribution of Key EHR Features", os.path.join(d_out, 'ehr_histogram_kde.png'), "Distribution of fundamental clinical biomarkers.", "EHR")
    plot_violin(df, ['chol', 'trestbps', 'thalch', 'oldpeak'], 'num', "EHR Feature Variation Across Disease Classes", os.path.join(d_out, 'ehr_violin_plot.png'), "Illustrates how important biomarkers vary across disease burden.", "EHR")
    plot_scatter(df, 'age', 'oldpeak', 'num', "Age vs ST Depression by Disease Class", os.path.join(d_out, 'ehr_scatter.png'), "Scatter relationship between age and exertion-induced ischemia proxy.", "EHR")
    plot_pca_projection(df, features, 'num', "PCA Projection of EHR Samples", os.path.join(d_out, 'ehr_projection.png'), "Visualizes whether samples separate in lower-dimensional feature space.", "EHR")
    return df

def generate_ecg():
    logging.info("Generating ECG visualizations...")
    df = pd.read_csv(ECG_PATH) if os.path.exists(ECG_PATH) else None
    features = ['heart_rate', 'rr_mean', 'rr_std', 'rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio', 'rhythm_stability', 'abnormality_score']
    df = clean_dataframe(df, target_col='severity_group', num_cols=features)
    if df is None: return None
    
    d_out = os.path.join(OUT_DIR, 'ecg')
    
    plot_class_distribution(df, 'severity_group', "ECG Severity Group Distribution", os.path.join(d_out, 'ecg_class_distribution.png'), "Shows how samples are distributed across severity categories.", "ECG")
    plot_histogram_kde(df, ['heart_rate', 'rmssd', 'lf_hf_ratio', 'abnormality_score'], "Distribution of Key ECG Features", os.path.join(d_out, 'ecg_histogram_kde.png'), "Distribution of electrophysiological metrics.", "ECG")
    plot_violin(df, ['rmssd', 'sdnn', 'lf_hf_ratio', 'rhythm_stability'], 'severity_group', "ECG Feature Variation Across Severity Groups", os.path.join(d_out, 'ecg_violin_plot.png'), "Illustrates how important biomarkers vary across disease burden.", "ECG")
    plot_scatter(df, 'lf_hf_ratio', 'abnormality_score', 'severity_group', "LF/HF Ratio vs ECG Abnormality Score", os.path.join(d_out, 'ecg_scatter.png'), "Sympathovagal balance vs global temporal signal distortion.", "ECG")
    plot_pca_projection(df, features, 'severity_group', "PCA Projection of ECG Samples", os.path.join(d_out, 'ecg_projection.png'), "Visualizes whether samples separate in lower-dimensional feature space.", "ECG")
    return df

def generate_mri():
    logging.info("Generating MRI visualizations...")
    df = pd.read_csv(MRI_PATH) if os.path.exists(MRI_PATH) else None
    features = ['lv_area', 'lv_diameter', 'rv_area', 'heart_area', 'heart_circularity', 'heart_aspect_ratio', 'global_intensity_mean', 'global_intensity_std', 'global_entropy', 'severity_score']
    df = clean_dataframe(df, target_col='severity_group', num_cols=features)
    if df is None: return None
    
    d_out = os.path.join(OUT_DIR, 'mri')
    
    plot_class_distribution(df, 'severity_group', "MRI Severity Group Distribution", os.path.join(d_out, 'mri_class_distribution.png'), "Shows how samples are distributed across severity categories.", "MRI")
    plot_histogram_kde(df, ['lv_area', 'heart_area', 'global_entropy', 'severity_score'], "Distribution of Key MRI Features", os.path.join(d_out, 'mri_histogram_kde.png'), "Distribution of cardiac structural and textural quantifications.", "MRI")
    plot_violin(df, ['lv_area', 'heart_area', 'heart_circularity', 'global_entropy'], 'severity_group', "MRI Feature Variation Across Severity Groups", os.path.join(d_out, 'mri_violin_plot.png'), "Illustrates how important biomarkers vary across disease burden.", "MRI")
    plot_scatter(df, 'heart_area', 'severity_score', 'severity_group', "Heart Area vs MRI Severity Score", os.path.join(d_out, 'mri_scatter.png'), "Structural dilation proxy plotted against continuous severity formulation.", "MRI")
    plot_pca_projection(df, features, 'severity_group', "PCA Projection of MRI Samples", os.path.join(d_out, 'mri_projection.png'), "Visualizes whether samples separate in lower-dimensional feature space.", "MRI")
    return df

def generate_multimodal(df_ehr, df_ecg, df_mri):
    """
    Multimodal visualizations are based on prototype-level fused representations for 
    methodological exploration and not direct subject-level clinical alignment.
    Datasets are truncated to minimum common length randomly to simulate a fused cohort space.
    """
    logging.info("Generating Multimodal prototype visualizations...")
    if df_ehr is None or df_ecg is None or df_mri is None: 
        logging.warning("Missing one of the datasets. Cannot generate multimodal plots.")
        return
        
    # Minimum common length for blind horizontal concatenation
    min_len = min(len(df_ehr), len(df_ecg), len(df_mri))
    ehr_trunc = df_ehr.sample(min_len, random_state=42).reset_index(drop=True)
    ecg_trunc = df_ecg.sample(min_len, random_state=42).reset_index(drop=True)
    mri_trunc = df_mri.sample(min_len, random_state=42).reset_index(drop=True)
    
    df_fused = pd.concat([ehr_trunc[['oldpeak', 'num']], 
                          ecg_trunc[['abnormality_score', 'severity_group']].rename(columns={'severity_group':'ecg_sev'}), 
                          mri_trunc[['heart_area', 'global_entropy', 'severity_group']].rename(columns={'severity_group':'mri_sev'})
                         ], axis=1)
    
    # We will use 'num' acting as the prototype fused label just for colorization
    df_fused['prototype_severity'] = df_fused['num']
    
    d_out = os.path.join(OUT_DIR, 'multimodal')
    
    # A. Grouped bar chart
    plt.figure(figsize=(8, 5))
    melted = df_fused[['num', 'ecg_sev', 'mri_sev']].melt(var_name='Modality', value_name='Class')
    sns.countplot(data=melted, x='Class', hue='Modality', palette='deep')
    plt.title("Severity Distribution Across Modalities", pad=15)
    plt.figtext(0.5, -0.05, "NOTE: Represents independent label distributions. Not subject-aligned.", ha="center", fontsize=9, color='gray')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(d_out, 'multimodal_class_comparison.png'), bbox_inches='tight')
    plt.close()
    register_plot("Multimodal", "Severity Distribution Across Modalities", os.path.join(d_out, 'multimodal_class_comparison.png'), "Compare distributions of raw label classes per isolated modality.")

    # B. Histogram KDE
    plot_histogram_kde(df_fused, ['oldpeak', 'abnormality_score', 'heart_area', 'global_entropy'], 
                       "Distribution of Representative Multimodal Features", 
                       os.path.join(d_out, 'multimodal_histogram_kde.png'), 
                       "Cross-modality feature distributions.", "Multimodal")

    # C. Violin
    plot_violin(df_fused, ['oldpeak', 'abnormality_score', 'heart_area', 'global_entropy'], 'prototype_severity',
                "Prototype Multimodal Feature Variation", os.path.join(d_out, 'multimodal_violin_plot.png'),
                "Illustrates how important biomarkers vary across disease burden.", "Multimodal")

    # D. Scatter
    plot_scatter(df_fused, 'abnormality_score', 'heart_area', 'prototype_severity',
                 "ECG Instability vs MRI Structural Burden", os.path.join(d_out, 'multimodal_scatter.png'),
                 "Highlights cross-feature relationships associated with structural or electrophysiological burden.", "Multimodal")

    # E. Projection
    plot_pca_projection(df_fused, ['oldpeak', 'abnormality_score', 'heart_area', 'global_entropy'], 'prototype_severity',
                        "Projection of Prototype Multimodal Feature Space", os.path.join(d_out, 'multimodal_projection.png'),
                        "Visualizes whether samples separate in lower-dimensional feature space.", "Multimodal")

def generate_summaries():
    sum_dir = os.path.join(OUT_DIR, 'summaries')
    json_path = os.path.join(sum_dir, 'visualization_manifest.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(manifest, indent=2))
        
    md_path = os.path.join(sum_dir, 'eda_summary.md')
    with open(md_path, 'w') as f:
        f.write("# CardioTwin Exploratory Data Analysis Summary\n\n")
        f.write("> **Disclaimer:** Multimodal visualizations are based on prototype-level fused representations for methodological exploration and not direct subject-level clinical alignment.\n\n")
        
        for struct in ['EHR', 'ECG', 'MRI', 'Multimodal']:
            f.write(f"## {struct} Dataset Analysis\n")
            for plot in manifest:
                if plot['modality'] == struct:
                    f.write(f"### {plot['title']}\n")
                    f.write(f"- **Path:** `{plot['path']}`\n")
                    f.write(f"- **Significance:** {plot['description']}\n\n")

if __name__ == "__main__":
    import json
    df_ehr = generate_ehr()
    df_ecg = generate_ecg()
    df_mri = generate_mri()
    generate_multimodal(df_ehr, df_ecg, df_mri)
    generate_summaries()
    logging.info("All visualizations generated successfully.")
