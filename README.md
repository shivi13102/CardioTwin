# CardioTwin: A Multimodal Cardiac Digital Twin Prototype

CardioTwin is an open-source framework for patient-specific cardiac digital twin modeling, integrating 4D MRI velocity fields, ECG signals, and physics-constrained simulations to enable longitudinal heart disease progression forecasting beyond traditional ML classification.

It implements **Cardiac-FM**, a multimodal foundation model that fuses Clinical (EHR), Electrical (ECG), and Structural (MRI) data into a shared digital twin state. It features a fast PyTorch backend and a premium React dashboard for disease progression simulation.

---

## 🚀 Quick Start Guide

### 1. Setup Backend (PyTorch + FastAPI)
The backend loads the trained `Cardiac-FM` model and serves the `/predict` and `/simulate` endpoints.

1. Open a terminal and navigate to the project root:
   ```bash
   cd "e:/SEMESTER SUBJECTS/6th SEMESTER/CARDIOTWIN"
   ```
2. Activate your virtual environment (if used):
   ```bash
   # Windows
   .\venv\Scripts\activate
   ```
3. Install required backend packages (if not already installed):
   ```bash
   pip install torch pandas numpy scikit-learn fastapi uvicorn shap
   ```
4. Run the model training script (if you haven't yet, to generate `models/cardiac_fm.pth`):
   ```bash
   python src/training/train_cardiac_fm.py
   ```
5. Start the FastAPI server:
   ```bash
   uvicorn api.main:app --reload
   ```
   > The API will run at `http://127.0.0.1:8000`. Keep this terminal open.

---

### 2. Setup Frontend (React + Vite)
The frontend is a modern, responsive doctor-facing dashboard built with Tailwind CSS, shadcn/ui, and Recharts.

1. Open a **new** terminal and navigate to the `frontend` folder:
   ```bash
   cd "e:/SEMESTER SUBJECTS/6th SEMESTER/CARDIOTWIN/frontend"
   ```
2. Install all Node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   > The dashboard will run at `http://localhost:5173`.

---

## 💡 How to Use the Dashboard

1. **Dashboard Home**: Explains the system capabilities and architecture. You can click Quick Actions to load sample patients.
2. **Patient Input**: Enter EHR, ECG, and MRI features manually or click one of the `Presets` (Low, Moderate, High risk). Click *Generate Digital Twin*.
3. **Twin Analysis**: 
   - View your baseline Risk Category (Low, Moderate, High) and Risk Score.
   - See how each modality contributes (Clinical, Electrical, Structural).
   - View the internal 16-D Twin State representation.
4. **Simulation Panel**: 
   - Click *Run Simulation*.
   - Adjust the deltas (e.g., increase Cholesterol by `+20`, decrease LVEF by `-10%`).
   - Click *Simulate Worsening* to see the projected Risk and Progression trend change.

## 🛠️ Architecture Overview
* **src/models/cardiac_fm.py**: The PyTorch neural network combining 3 modality encoders and a fusion block.
* **src/training/train_cardiac_fm.py**: Generates the model weights and data scalers.
* **api/main.py**: The FastAPI interface mapping raw inputs to model tensors.
* **frontend/src/**: The Vite React codebase with framer-motion animations and Recharts tracking.

Of course — here is your text written properly, keeping the content the same and not changing anything:

---

# **Integration of Rule-Guided Scoring with Cardiac-FM for Digital Twin-Based Cardiac Risk and Progression Modeling**

## **1. Overview**

The proposed CardioTwin framework is designed as a multimodal digital twin prototype for cardiac risk analysis and disease progression assessment. The system combines clinical information from Electronic Health Records (EHR), electrophysiological information from Electrocardiogram (ECG) features, and structural-functional information from Cardiac MRI-derived features. These multimodal inputs are processed jointly to construct a patient-specific representation of cardiac state.

At the core of the framework lies Cardiac-FM, a foundation-style multimodal fusion model. However, rather than relying only on opaque end-to-end predictions, the framework incorporates a structured scoring layer that transforms clinically relevant inputs into interpretable branch-wise burden scores and disease progression indicators. This design allows the system to preserve the predictive strength of multimodal learning while also producing outputs that are explainable, stable, and suitable for digital twin-style simulation.

Thus, the final CardioTwin decision engine operates through a hybrid collaboration between:

* a multimodal machine learning model that learns latent patient representations, and
* an interpretable formula-based scoring layer that converts patient features into clinically meaningful severity, contribution, and progression metrics.

This hybrid structure is especially appropriate for a prototype setting, where interpretability and presentation clarity are essential alongside predictive capability.

---

## **2. Role of Cardiac-FM in the Framework**

### **2.1 Multimodal latent representation learning**

Cardiac-FM is responsible for learning a shared patient-level latent representation from heterogeneous modalities. Each modality is first encoded independently:

* the EHR encoder captures clinical burden and demographic-context information,
* the ECG encoder captures electrical instability and rhythm-related dysfunction,
* the MRI encoder captures structural remodeling, ventricular dysfunction, and tissue-level abnormality.

If the three input modalities are denoted by:

**x_ehr, x_ecg, x_mri**

then the corresponding encoder outputs are:

**z_ehr = f_ehr(x_ehr)**
**z_ecg = f_ecg(x_ecg)**
**z_mri = f_mri(x_mri)**

These latent embeddings are then fused:

**z_twin = f_fusion(z_ehr, z_ecg, z_mri)**

where **z_twin** is the digital twin state, representing a compact patient-specific cardiac state in multimodal latent space.

### **2.2 Why Cardiac-FM alone is not sufficient for the prototype**

Although Cardiac-FM can generate powerful latent representations and overall risk outputs, direct use of raw model activations or uncalibrated branch outputs can lead to poor interpretability. In early prototype behavior, this resulted in unstable or saturated outputs such as all modality scores being displayed as 100%, even when the overall category was only moderate.

To address this, an explicit formula-driven scoring layer was introduced. This layer collaborates with the model by translating raw multimodal inputs into bounded, interpretable scores that can be displayed in the frontend and used in simulation.

Therefore, in the current design:

* Cardiac-FM provides multimodal representation learning and fusion intelligence
* The scoring layer provides stable burden scores, progression estimation, and disease stage interpretation

---

## **3. Motivation for the Formula-Based Scoring Layer**

The scoring layer was introduced for five main reasons.

### **3.1 Interpretability**

Clinical applications require outputs that are understandable to non-technical users. A doctor-facing digital twin dashboard should display meaningful quantities such as:

* clinical burden,
* electrical instability,
* structural dysfunction,
* progression tendency,
* disease stage.

These are easier to communicate when they are derived through transparent mathematical formulations.

### **3.2 Numerical stability**

Raw latent vectors from neural models are often unbounded and difficult to display directly. A normalized scoring layer ensures that all displayed quantities lie in a bounded interval, typically **[0,1]**, and can be safely converted into percentages for the dashboard.

### **3.3 Simulation support**

Digital twin simulation requires deterministic response to perturbations. If cholesterol is increased or LVEF is reduced, the system should update risk and progression in a predictable manner. The formula-based layer supports this by explicitly tying each score to clinically relevant inputs.

### **3.4 Clinical alignment**

Even though the formulas are not clinically validated medical scores, they reflect intuitive domain logic:

* higher age, cholesterol, ST depression, vessel burden, and thallium abnormality should increase clinical risk,
* higher arrhythmia burden and lower rhythm stability should increase electrical risk,
* lower ejection fraction and higher wall motion abnormality should increase structural dysfunction.

### **3.5 Prototype extensibility**

The formula-based layer can later be replaced or calibrated using learned score heads, but for the current prototype it provides a robust bridge between machine learning outputs and human-readable decision support.

---

## **4. Multimodal Inputs Used in CardioTwin**

The scoring framework operates on four groups of inputs.

### **4.1 Clinical inputs**

These represent patient-level risk factors and stress-related cardiac behavior:

* age
* sex
* chest pain type (**cp**)
* resting blood pressure (**trestbps**)
* serum cholesterol (**chol**)
* fasting blood sugar (**fbs**)
* resting ECG category (**restecg**)
* maximum heart rate achieved (**thalach**)
* exercise-induced angina (**exang**)
* exercise ST depression (**oldpeak**)
* slope of ST segment (**slope**)
* number of major vessels (**ca**)
* thallium stress result (**thal**)
* clinical risk group

### **4.2 Electrical inputs**

These represent heart rhythm variability and electrical instability:

* **std_rr**
* **mean_rr**
* **rmssd**
* **pnn50**
* low-frequency power
* high-frequency power
* LF/HF spectral ratio
* P-wave duration
* QRS duration
* T-wave amplitude
* ST segment elevation
* rhythm stability score
* arrhythmia burden
* abnormality score

### **4.3 Structural inputs**

These represent ventricular size, cardiac function, and tissue-level abnormality:

* LVEDV, LVESV
* RVEDV, RVESV
* LVEF, RVEF
* LV stroke volume
* LV mass
* LV wall thickness
* heart eccentricity
* LV area
* myocardial strain
* wall motion score
* dysfunction score
* structural abnormality

### **4.4 Fusion inputs**

These represent cross-modal derived severity signals:

* **fusion_clinical_ecg**
* **fusion_ecg_mri**
* **fusion_severity_index**

These fusion variables help capture interactions between branch-level abnormalities.

---

## **5. Normalization Strategy**

Since the raw variables come from different scales and units, all branch calculations begin with normalization. A min-max bounded transform is used:

**norm(x; a, b) = clip((x - a) / (b - a), 0, 1)**

where:

**clip(v, 0, 1) = min(1, max(0, v))**

For inverse-risk features, where lower values indicate worse health, the inverse normalization is applied:

**invnorm(x; a, b) = 1 - norm(x; a, b)**

This ensures that all normalized variables increase in the direction of increasing risk.

For example:

* higher cholesterol should increase risk, so regular normalization is used;
* lower maximum heart rate and lower ejection fraction indicate worse condition, so inverse normalization is used.

This common normalization space allows heterogeneous variables to be aggregated meaningfully.

---

## **6. Clinical Score Formulation**

### **6.1 Purpose**

The Clinical Score quantifies patient burden from EHR and symptom-related inputs. It represents the degree of conventional cardiovascular risk burden and exercise-associated abnormality.

### **6.2 Normalized clinical terms**

The following terms are defined:

**age_n = norm(age; 30, 80)**
**cp_n = (3 - cp) / 3**
**trestbps_n = norm(trestbps; 90, 180)**
**chol_n = norm(chol; 150, 350)**
**fbs_n = fbs**
**restecg_n = restecg / 2**
**thalach_n = invnorm(thalach; 60, 190)**
**exang_n = exang**
**oldpeak_n = norm(oldpeak; 0, 4)**
**slope_n = (2 - slope) / 2**
**ca_n = ca / 3**
**thal_n = (thal - 1) / 2**
**group_n = clinical_risk_group / 2**
**sex_n = sex**

### **6.3 Weighted aggregation**

The final Clinical Score is computed as:

**ClinicalScore = 0.08age_n + 0.03sex_n + 0.10cp_n + 0.06trestbps_n + 0.06chol_n + 0.04fbs_n + 0.04restecg_n + 0.08thalach_n + 0.09exang_n + 0.12oldpeak_n + 0.07slope_n + 0.10ca_n + 0.06thal_n + 0.07group_n**

The result is clipped to **[0,1]**.

### **6.4 Interpretation**

A higher Clinical Score indicates greater conventional cardiovascular burden. Elevated score values are driven by combinations such as:

* advanced age,
* poor chest pain pattern,
* elevated cholesterol and blood pressure,
* exercise-induced angina,
* significant ST depression,
* abnormal thallium test,
* higher vessel involvement.

---

## **7. Electrical Score Formulation**

### **7.1 Purpose**

The Electrical Score quantifies instability in the electrophysiological state of the heart. It combines variability, conduction, spectral imbalance, rhythm regularity, and arrhythmic burden.

### **7.2 Normalized electrical terms**

**stdrr_n = norm(std_rr; 0.02, 0.15)**
**meanrr_n = clip(|mean_rr - 0.80| / 0.40, 0, 1)**
**rmssd_n = invnorm(rmssd; 5, 40)**
**pnn50_n = invnorm(pnn50; 0, 20)**
**ratio_n = norm(spectral_ratio; 1, 4)**
**qrs_n = norm(qrs_duration; 80, 160)**
**st_n = norm(st_segment_elevation; 0, 0.30)**
**rhythm_n = 1 - rhythm_stability_score**
**arr_n = arrhythmia_burden**
**abn_n = abnormality_score**
**hf_n = invnorm(high_freq; 0.05, 0.30)**

### **7.3 Weighted aggregation**

**ElectricalScore = 0.07stdrr_n + 0.05meanrr_n + 0.08rmssd_n + 0.05pnn50_n + 0.07ratio_n + 0.08qrs_n + 0.05st_n + 0.15rhythm_n + 0.20arr_n + 0.15abn_n + 0.05hf_n**

The result is clipped to **[0,1]**.

### **7.4 Interpretation**

Higher Electrical Score indicates more severe electrical instability. Important drivers include:

* high arrhythmia burden,
* low rhythm stability,
* low vagal HRV markers,
* wider QRS,
* elevated ST deviation,
* adverse spectral imbalance.

---

## **8. Structural Score Formulation**

### **8.1 Purpose**

The Structural Score summarizes structural and functional cardiac abnormality derived from MRI-related features. It captures chamber enlargement, reduced ventricular function, abnormal myocardial mechanics, and structural remodeling.

### **8.2 Normalized structural terms**

**lvedv_n = norm(lvedv; 100, 220)**
**lvesv_n = norm(lvesv; 30, 150)**
**rvedv_n = norm(rvedv; 100, 220)**
**rvesv_n = norm(rvesv; 30, 140)**
**lvef_n = invnorm(lvef; 25, 60)**
**rvef_n = invnorm(rvef; 25, 55)**
**lvsv_n = invnorm(lvsv; 40, 90)**
**lvmass_n = norm(lv_mass; 100, 250)**
**wallthick_n = norm(lv_wall_thickness; 8, 16)**
**ecc_n = norm(heart_eccentricity; 0.4, 0.8)**
**lvarea_n = norm(lv_area; 20, 40)**
**strain_n = clip((20 - |myocardial_strain|) / 14, 0, 1)**
**wallmotion_n = (wall_motion_score - 1) / 2**
**dys_n = dysfunction_score**
**struct_n = structural_abnormality**

### **8.3 Weighted aggregation**

**StructuralScore = 0.04lvedv_n + 0.06lvesv_n + 0.03rvedv_n + 0.03rvesv_n + 0.16lvef_n + 0.04rvef_n + 0.04lvsv_n + 0.07lvmass_n + 0.05wallthick_n + 0.05ecc_n + 0.03lvarea_n + 0.10strain_n + 0.10wallmotion_n + 0.10dys_n + 0.10struct_n**

The final value is clipped to **[0,1]**.

### **8.4 Interpretation**

Higher Structural Score corresponds to more severe structural remodeling and ventricular dysfunction. Large contributions arise from:

* lower ejection fraction,
* poorer strain,
* higher wall motion abnormality,
* increased dysfunction score,
* increased structural abnormality,
* adverse chamber and mass measurements.

---

## **9. Global Risk Score**

### **9.1 Purpose**

The Global Risk Score integrates the three modality-specific burden scores with fusion features to yield an overall multimodal cardiac risk estimate.

### **9.2 Formula**

**GlobalRisk = 0.28ClinicalScore + 0.24ElectricalScore + 0.28StructuralScore + 0.07fusion_clinical_ecg + 0.05fusion_ecg_mri + 0.08fusion_severity_index**

The result is clipped to **[0,1]**.

### **9.3 Interpretation**

This score reflects a comprehensive multimodal disease burden. The branch weights ensure that no single modality fully dominates the decision, while the fusion terms allow the system to account for interactions across modalities.

---

## **10. Progression Score**

### **10.1 Purpose**

The Progression Score estimates the likelihood of worsening disease trajectory. Unlike the Global Risk Score, which focuses on current burden, the Progression Score emphasizes features more strongly linked to deterioration.

### **10.2 Formula**

**ProgressionScore = 0.16oldpeak_n + 0.10thalach_n + 0.10arr_n + 0.10abn_n + 0.14lvef_n + 0.10wallmotion_n + 0.10dys_n + 0.10struct_n + 0.05fusion_ecg_mri + 0.05fusion_severity_index**

The result is clipped to **[0,1]**.

### **10.3 Interpretation**

This score is intended to capture worsening tendency. It rises in the presence of:

* poor exercise response,
* increased electrical abnormality,
* reduced ventricular function,
* high wall motion abnormality,
* structural dysfunction,
* unfavorable cross-modal severity coupling.

---

## **11. Risk Category Assignment**

The Global Risk Score is mapped into a discrete categorical interpretation:

* **Low** if **GlobalRisk < 0.40**
* **Moderate** if **0.40 ≤ GlobalRisk < 0.65**
* **High** if **GlobalRisk ≥ 0.65**

This categorical conversion allows the frontend to show clinician-friendly classes while still preserving continuous underlying scores.

---

## **12. Risk Probabilities**

Instead of displaying only a single label, the framework also provides soft risk probabilities for Low, Moderate, and High.

### **12.1 Logits**

**L_low = -4 · GlobalRisk**
**L_moderate = 2 - 8 · |GlobalRisk - 0.55|**
**L_high = 4 · GlobalRisk - 2**

### **12.2 Softmax**

**P_i = e^(L_i) / (e^(L_low) + e^(L_moderate) + e^(L_high))**

where **i ∈ {low, moderate, high}**.

These probabilities sum to 1 and can be used for the frontend risk probability chart.

---

## **13. Modality Contribution Breakdown**

### **13.1 Purpose**

The branch scores describe severity, but the frontend also needs to communicate which modality contributes most strongly to the final decision.

### **13.2 Raw contributions**

**C_clinical = 0.28ClinicalScore + 0.07fusion_clinical_ecg**
**C_electrical = 0.24ElectricalScore + 0.04fusion_clinical_ecg + 0.05fusion_ecg_mri**
**C_structural = 0.28StructuralScore + 0.03fusion_ecg_mri + 0.08fusion_severity_index**

### **13.3 Normalization**

**ContributionSum = C_clinical + C_electrical + C_structural**

**ClinicalContribution = C_clinical / ContributionSum**
**ElectricalContribution = C_electrical / ContributionSum**
**StructuralContribution = C_structural / ContributionSum**

These values sum to 1 and are displayed as percentages.

### **13.4 Interpretation**

This contribution breakdown is not the same as severity. A modality may have a high burden score, but its relative contribution to final decision can still differ depending on fusion terms and the weighting scheme.

---

## **14. Disease Progress Index**

### **14.1 Purpose**

A single disease progress indicator was introduced for the frontend to summarize both current disease burden and anticipated progression tendency.

### **14.2 Formula**

**DiseaseProgressIndex = 0.6 · GlobalRisk + 0.4 · ProgressionScore**

The final value is clipped to **[0,1]**.

### **14.3 Interpretation**

This provides a compact view of the patient’s overall status along a disease continuum.

---

## **15. Disease Stage Mapping**

The Disease Progress Index is translated into a stage label:

* **0.00 – 0.25:** Minimal burden
* **0.26 – 0.50:** Early disease
* **0.51 – 0.70:** Established disease
* **0.71 – 0.85:** Advanced disease
* **0.86 – 1.00:** Critical progression

This stage label allows the digital twin to present disease progression in a clinically intuitive form.

---

## **16. Collaboration Between Cardiac-FM and the Scoring Layer**

### **16.1 Parallel roles**

The final system should not be interpreted as “machine learning versus formulas.” Instead, both are working together.

**Cardiac-FM contributes:**

* multimodal encoding,
* latent state construction,
* flexible fusion,
* future extensibility,
* ability to support true learned inference later.

**The formula-based layer contributes:**

* stable and bounded outputs,
* branch-wise interpretability,
* simulation transparency,
* stage mapping,
* doctor-facing summary generation.

### **16.2 Practical collaboration**

In the current prototype, Cardiac-FM defines the architecture and multimodal representation pipeline, while the scoring layer acts as a calibrated decision-support surface on top of the patient inputs and fusion context.

This means the digital twin can be interpreted at two levels:

* latent level: the hidden twin state learned by Cardiac-FM, and
* clinical dashboard level: the interpretable branch scores, risk score, progression, contribution, and stage.

### **16.3 Why this hybrid design is valuable**

A purely black-box deep fusion model would be harder to debug and explain. A purely handcrafted rule-based system would lack expressive multimodal modeling capability. The proposed hybrid design balances both needs:

* it is sufficiently intelligent for multimodal fusion,
* yet sufficiently interpretable for a prototype digital twin application.

---

## **17. Doctor-Style Summary Generation**

The final dashboard also generates a doctor-style narrative summary. This is produced using computed outputs rather than hardcoded text.

The summary depends on:

* Risk Category,
* Disease Stage,
* Progression Score,
* dominant modality among Clinical, Electrical, and Structural Scores,
* contribution pattern.

For example:

* if Structural Score is dominant and risk is high, the summary emphasizes structural dysfunction and ventricular impairment;
* if Electrical Score is dominant, the summary highlights rhythm instability and arrhythmic concern;
* if Clinical Score is dominant at moderate risk, the summary emphasizes elevated conventional clinical burden and recommends follow-up.

This narrative converts numeric model outputs into interpretable decision support.

---

## **18. Example Interpretation for a High-Risk Sample**

For a representative high-risk patient, the framework may generate:

* Clinical Score: 78%
* Electrical Score: 66%
* Structural Score: 65%
* Global Risk Score: 71%
* Progression Score: 67%
* Disease Progress Index: 69%
* Disease Stage: Established disease, nearing advanced burden

The associated summary may read:

*Predicted high risk is driven by substantial clinical burden, marked electrical instability, and significant structural dysfunction. Progression tendency is elevated, with reduced ventricular function, abnormal wall motion, and arrhythmic burden contributing strongly. Current findings suggest established cardiac disease burden approaching advanced stage. Close cardiology follow-up is advised.*

This style of output is more useful for a general-doctor dashboard than raw model logits.

---

## **19. Advantages of the Proposed Design**

The integration of formula-guided burden scoring with Cardiac-FM offers several benefits:

### **19.1 Interpretability**

The system provides branch-wise reasoning instead of only a final label.

### **19.2 Stability**

All displayed outputs are bounded and numerically stable.

### **19.3 Modular design**

The scoring layer can later be replaced or calibrated without redesigning the entire model.

### **19.4 Digital twin suitability**

The explicit dependence on physiologically meaningful variables makes the framework well-suited for “what-if” simulation.

### **19.5 Presentation value**

For academic demonstration, the hybrid framework is much easier to explain than a fully opaque fusion model.

---

## **20. Limitations**

It is important to state the limitations clearly.

* These formulas are prototype scoring rules, not validated medical scoring guidelines.
* The displayed burden scores are surrogate indicators, not direct clinical biomarkers.
* The disease progression mechanism is an approximation, not a true longitudinal physiological simulation.
* The fusion weights and score thresholds are designed for prototype interpretability and may require recalibration on real multimodal patient cohorts.
* If multimodal alignment is synthetic rather than patient-wise, the resulting scores must be interpreted cautiously.

---

## **21. Conclusion**

The CardioTwin framework combines the multimodal representation power of Cardiac-FM with an interpretable rule-guided scoring layer to produce clinically understandable digital twin outputs. Clinical, electrical, and structural features are first normalized and transformed into branch-level burden scores. These are then integrated with fusion-derived severity markers to produce a Global Risk Score, Progression Score, Disease Progress Index, and Disease Stage. Finally, these outputs are summarized in doctor-style text for decision support.

This design enables the system to behave not only as a multimodal classifier, but as a prototype cardiac digital twin that supports:

* patient state representation,
* multimodal burden estimation,
* disease progress visualization,
* what-if simulation,
* and interpretable clinical communication.

The result is a practical and research-appropriate bridge between machine learning and digital twin-based precision cardiology.
