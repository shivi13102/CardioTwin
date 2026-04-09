# CardioTwin Exploratory Data Analysis Summary

> **Disclaimer:** Multimodal visualizations are based on prototype-level fused representations for methodological exploration and not direct subject-level clinical alignment.

## EHR Dataset Analysis
### EHR Disease Class Distribution
- **Path:** `/visualization/ehr/ehr_class_distribution.png`
- **Significance:** Shows how samples are distributed across severity categories.

### Distribution of Key EHR Features
- **Path:** `/visualization/ehr/ehr_histogram_kde.png`
- **Significance:** Distribution of fundamental clinical biomarkers.

### EHR Feature Variation Across Disease Classes
- **Path:** `/visualization/ehr/ehr_violin_plot.png`
- **Significance:** Illustrates how important biomarkers vary across disease burden.

### Age vs ST Depression by Disease Class
- **Path:** `/visualization/ehr/ehr_scatter.png`
- **Significance:** Scatter relationship between age and exertion-induced ischemia proxy.

### PCA Projection of EHR Samples
- **Path:** `/visualization/ehr/ehr_projection.png`
- **Significance:** Visualizes whether samples separate in lower-dimensional feature space.

## ECG Dataset Analysis
### ECG Severity Group Distribution
- **Path:** `/visualization/ecg/ecg_class_distribution.png`
- **Significance:** Shows how samples are distributed across severity categories.

### Distribution of Key ECG Features
- **Path:** `/visualization/ecg/ecg_histogram_kde.png`
- **Significance:** Distribution of electrophysiological metrics.

### ECG Feature Variation Across Severity Groups
- **Path:** `/visualization/ecg/ecg_violin_plot.png`
- **Significance:** Illustrates how important biomarkers vary across disease burden.

### LF/HF Ratio vs ECG Abnormality Score
- **Path:** `/visualization/ecg/ecg_scatter.png`
- **Significance:** Sympathovagal balance vs global temporal signal distortion.

### PCA Projection of ECG Samples
- **Path:** `/visualization/ecg/ecg_projection.png`
- **Significance:** Visualizes whether samples separate in lower-dimensional feature space.

## MRI Dataset Analysis
### MRI Severity Group Distribution
- **Path:** `/visualization/mri/mri_class_distribution.png`
- **Significance:** Shows how samples are distributed across severity categories.

### Distribution of Key MRI Features
- **Path:** `/visualization/mri/mri_histogram_kde.png`
- **Significance:** Distribution of cardiac structural and textural quantifications.

### MRI Feature Variation Across Severity Groups
- **Path:** `/visualization/mri/mri_violin_plot.png`
- **Significance:** Illustrates how important biomarkers vary across disease burden.

### Heart Area vs MRI Severity Score
- **Path:** `/visualization/mri/mri_scatter.png`
- **Significance:** Structural dilation proxy plotted against continuous severity formulation.

### PCA Projection of MRI Samples
- **Path:** `/visualization/mri/mri_projection.png`
- **Significance:** Visualizes whether samples separate in lower-dimensional feature space.

## Multimodal Dataset Analysis
### Severity Distribution Across Modalities
- **Path:** `/visualization/multimodal/multimodal_class_comparison.png`
- **Significance:** Compare distributions of raw label classes per isolated modality.

### Distribution of Representative Multimodal Features
- **Path:** `/visualization/multimodal/multimodal_histogram_kde.png`
- **Significance:** Cross-modality feature distributions.

### Prototype Multimodal Feature Variation
- **Path:** `/visualization/multimodal/multimodal_violin_plot.png`
- **Significance:** Illustrates how important biomarkers vary across disease burden.

### ECG Instability vs MRI Structural Burden
- **Path:** `/visualization/multimodal/multimodal_scatter.png`
- **Significance:** Highlights cross-feature relationships associated with structural or electrophysiological burden.

### Projection of Prototype Multimodal Feature Space
- **Path:** `/visualization/multimodal/multimodal_projection.png`
- **Significance:** Visualizes whether samples separate in lower-dimensional feature space.

