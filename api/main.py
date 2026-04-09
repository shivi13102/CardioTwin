"""
CardioTwin FastAPI Backend
Serves the Cardiac-FM model for real-time inference and simulation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import joblib
import copy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.models.cardiac_fm import CardiacFM

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="CardioTwin API",
    description="Multimodal Cardiac Digital Twin — Powered by Cardiac-FM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
scalers = joblib.load(MODELS_DIR / "cardiac_fm_scalers.pkl")

CLINICAL_COLS = scalers["clinical_cols"]   # e.g. clinical_age, clinical_sex, ...
ECG_COLS = scalers["ecg_cols"]             # ecg_0 .. ecg_9
MRI_COLS = scalers["mri_cols"]             # mri_0 .. mri_14

EHR_DIM = len(CLINICAL_COLS)
ECG_DIM = len(ECG_COLS)
MRI_DIM = len(MRI_COLS)

device = torch.device("cpu")
cardiac_fm = CardiacFM(ehr_dim=EHR_DIM, ecg_dim=ECG_DIM, mri_dim=MRI_DIM)
cardiac_fm.load_state_dict(torch.load(MODELS_DIR / "cardiac_fm.pth", map_location=device))
cardiac_fm.eval()

RISK_LABELS = ["Low", "Moderate", "High"]
RISK_COLORS = ["green", "amber", "red"]

# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────
class ClinicalFeatures(BaseModel):
    age: float = 55
    sex: float = 1
    cp: float = 0
    trestbps: float = 130
    chol: float = 230
    fbs: float = 0
    restecg: float = 0
    thalach: float = 150
    exang: float = 0
    oldpeak: float = 1.0
    slope: float = 1
    ca: float = 0
    thal: float = 2
    clinical_risk_group: float = 0  # 0=low,1=mod,2=high

class ECGFeatures(BaseModel):
    std_rr: float = 0.05
    mean_rr: float = 0.85
    rmssd: float = 30.0
    pnn50: float = 15.0
    low_freq: float = 0.3
    high_freq: float = 0.25
    spectral_ratio: float = 1.2
    p_wave_duration: float = 90.0
    qrs_duration: float = 100.0
    t_wave_amplitude: float = 0.25
    st_segment_elevation: float = 0.1
    rhythm_stability_score: float = 0.85
    arrhythmia_burden: float = 0.05
    abnormality_score: float = 0.1

class MRIFeatures(BaseModel):
    lvedv: float = 140.0
    lvesv: float = 55.0
    rvedv: float = 145.0
    rvesv: float = 60.0
    lvef: float = 60.0
    rvef: float = 58.0
    lvsv: float = 85.0
    lv_mass: float = 160.0
    lv_wall_thickness: float = 9.5
    heart_eccentricity: float = 0.5
    lv_area: float = 28.0
    myocardial_strain: float = -18.0
    wall_motion_score: float = 1.0
    dysfunction_score: float = 0.2
    structural_abnormality: float = 0.1

class PatientInput(BaseModel):
    clinical: ClinicalFeatures
    ecg: ECGFeatures
    mri: MRIFeatures

class SimulationDeltas(BaseModel):
    delta_chol: float = 0
    delta_thalach: float = 0
    delta_oldpeak: float = 0
    delta_lvef: float = 0
    delta_wall_motion_score: float = 0
    delta_structural_abnormality: float = 0
    delta_std_rr: float = 0
    delta_rmssd: float = 0

class SimulateRequest(BaseModel):
    patient: PatientInput
    deltas: SimulationDeltas
    n_steps: int = Field(default=5, ge=1, le=10)

# ─────────────────────────────────────────────
# Core Inference Helper
# ─────────────────────────────────────────────
def _clinical_to_vector(c: ClinicalFeatures) -> List[float]:
    """Map user-friendly clinical fields → internal clinical column order."""
    # Build a dict matching the CLINICAL_COLS from training:
    # [clinical_age, clinical_sex, clinical_dataset, clinical_cp, clinical_trestbps,
    #  clinical_chol, clinical_fbs, clinical_restecg, clinical_thalach, clinical_exang,
    #  clinical_oldpeak, clinical_slope, clinical_ca, clinical_thal,
    #  clinical_age_group, clinical_cholesterol_risk, clinical_bp_category,
    #  clinical_age_chol_interaction, clinical_max_hr_ratio, clinical_oldpeak_severity,
    #  clinical_abnormal_count, clinical_risk_group]
    age_group = 0 if c.age < 40 else (1 if c.age < 55 else (2 if c.age < 65 else 3))
    chol_risk  = 0 if c.chol < 200 else (1 if c.chol < 240 else 2)
    bp_cat     = 0 if c.trestbps < 120 else (1 if c.trestbps < 130 else (2 if c.trestbps < 140 else 3))
    age_chol   = c.age * c.chol / 1000
    max_hr_r   = c.thalach / max(220 - c.age, 1)
    op_sev     = 0 if c.oldpeak <= 0.5 else (1 if c.oldpeak <= 1.5 else 2)
    ab_count   = c.exang + c.fbs + op_sev

    mapping = {
        "clinical_age": c.age, "clinical_sex": c.sex, "clinical_dataset": 0,
        "clinical_cp": c.cp, "clinical_trestbps": c.trestbps, "clinical_chol": c.chol,
        "clinical_fbs": c.fbs, "clinical_restecg": c.restecg, "clinical_thalach": c.thalach,
        "clinical_exang": c.exang, "clinical_oldpeak": c.oldpeak, "clinical_slope": c.slope,
        "clinical_ca": c.ca, "clinical_thal": c.thal,
        "clinical_age_group": age_group, "clinical_cholesterol_risk": chol_risk,
        "clinical_bp_category": bp_cat, "clinical_age_chol_interaction": age_chol,
        "clinical_max_hr_ratio": max_hr_r, "clinical_oldpeak_severity": op_sev,
        "clinical_abnormal_count": ab_count, "clinical_risk_group": c.clinical_risk_group,
    }
    return [mapping.get(col, 0.0) for col in CLINICAL_COLS]

def _ecg_to_vector(e: ECGFeatures) -> List[float]:
    """Map ECG features → model's ecg_0..13 order."""
    vals = [e.std_rr, e.mean_rr, e.rmssd, e.pnn50,
            e.low_freq, e.high_freq, e.spectral_ratio,
            e.p_wave_duration, e.qrs_duration, e.t_wave_amplitude,
            e.st_segment_elevation, e.rhythm_stability_score,
            e.arrhythmia_burden, e.abnormality_score]
    # Model was trained on 14 ECG cols (ecg_0..13)
    return vals[:len(ECG_COLS)]

def _mri_to_vector(m: MRIFeatures) -> List[float]:
    """Map MRI features → model's mri_0..14 order."""
    vals = [m.lvedv, m.lvesv, m.rvedv, m.rvesv, m.lvef, m.rvef,
            m.lvsv, m.lv_mass, m.lv_wall_thickness, m.heart_eccentricity,
            m.lv_area, m.myocardial_strain, m.wall_motion_score,
            m.dysfunction_score, m.structural_abnormality]
    return vals[:len(MRI_COLS)]

import math

def clip(v, a, b):
    return max(a, min(b, v))

def norm(x, a, b):
    return clip((x - a) / (b - a) if b != a else 0, 0, 1)

def invnorm(x, a, b):
    return 1.0 - norm(x, a, b)

def _infer(c: ClinicalFeatures, e: ECGFeatures, m: MRIFeatures, baseline: dict = None) -> Dict[str, Any]:
    """Scale inputs, compute model vectors, and apply prototype deterministic heuristic formulas."""
    cv = _clinical_to_vector(c)
    ev = _ecg_to_vector(e)
    mv = _mri_to_vector(m)

    c_arr = scalers["clinical"].transform([cv])
    e_arr = scalers["ecg"].transform([ev])
    m_arr = scalers["mri"].transform([mv])

    t_c = torch.FloatTensor(c_arr)
    t_e = torch.FloatTensor(e_arr)
    t_m = torch.FloatTensor(m_arr)

    with torch.no_grad():
        out = cardiac_fm(t_c, t_e, t_m)

    twin_state = out["twin_state"][0].tolist()

    # Fusion variables extracted smoothly via sigmoid for boundedness without ML randomness
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    fusion_clinical_ecg = sigmoid(twin_state[0])
    fusion_ecg_mri = sigmoid(twin_state[1])
    fusion_severity_index = sigmoid(twin_state[2])

    # STEP 2: CLINICAL SCORE
    age_n = norm(c.age, 30, 80)
    cp_n = (3 - c.cp) / 3
    trestbps_n = norm(c.trestbps, 90, 180)
    chol_n = norm(c.chol, 150, 350)
    fbs_n = c.fbs
    restecg_n = c.restecg / 2
    thalach_n = invnorm(c.thalach, 60, 190)
    exang_n = c.exang
    oldpeak_n = norm(c.oldpeak, 0, 4)
    slope_n = (2 - c.slope) / 2
    ca_n = c.ca / 3
    thal_n = (c.thal - 1) / 2
    group_n = c.clinical_risk_group / 2
    sex_n = c.sex

    ClinicalScore = (
        0.08*age_n + 0.03*sex_n + 0.10*cp_n + 0.06*trestbps_n +
        0.06*chol_n + 0.04*fbs_n + 0.04*restecg_n + 0.08*thalach_n +
        0.09*exang_n + 0.12*oldpeak_n + 0.07*slope_n + 0.10*ca_n +
        0.06*thal_n + 0.07*group_n
    )
    ClinicalScore = clip(ClinicalScore, 0, 1)

    # STEP 3: ELECTRICAL SCORE
    stdrr_n = norm(e.std_rr, 0.02, 0.15)
    meanrr_n = clip(abs(e.mean_rr - 0.80) / 0.40, 0, 1)
    rmssd_n = invnorm(e.rmssd, 5, 40)
    pnn50_n = invnorm(e.pnn50, 0, 20)
    ratio_n = norm(e.spectral_ratio, 1, 4)
    qrs_n = norm(e.qrs_duration, 80, 160)
    st_n = norm(e.st_segment_elevation, 0, 0.30)
    rhythm_n = 1 - e.rhythm_stability_score
    arr_n = e.arrhythmia_burden
    abn_n = e.abnormality_score
    hf_n = invnorm(e.high_freq, 0.05, 0.30)

    ElectricalScore = (
        0.07*stdrr_n + 0.05*meanrr_n + 0.08*rmssd_n + 0.05*pnn50_n +
        0.07*ratio_n + 0.08*qrs_n + 0.05*st_n + 0.15*rhythm_n +
        0.20*arr_n + 0.15*abn_n + 0.05*hf_n
    )
    ElectricalScore = clip(ElectricalScore, 0, 1)

    # STEP 4: STRUCTURAL SCORE
    lvedv_n = norm(m.lvedv, 100, 220)
    lvesv_n = norm(m.lvesv, 30, 150)
    rvedv_n = norm(m.rvedv, 100, 220)
    rvesv_n = norm(m.rvesv, 30, 140)
    lvef_n = invnorm(m.lvef, 25, 60)
    rvef_n = invnorm(m.rvef, 25, 55)
    lvsv_n = invnorm(m.lvsv, 40, 90)
    lvmass_n = norm(m.lv_mass, 100, 250)
    wallthick_n = norm(m.lv_wall_thickness, 8, 16)
    ecc_n = norm(m.heart_eccentricity, 0.4, 0.8)
    lvarea_n = norm(m.lv_area, 20, 40)
    strain_n = clip((20 - abs(m.myocardial_strain)) / 14, 0, 1)
    wallmotion_n = (m.wall_motion_score - 1) / 2
    dys_n = m.dysfunction_score
    struct_n = m.structural_abnormality

    StructuralScore = (
        0.04*lvedv_n + 0.06*lvesv_n + 0.03*rvedv_n + 0.03*rvesv_n +
        0.16*lvef_n + 0.04*rvef_n + 0.04*lvsv_n + 0.07*lvmass_n +
        0.05*wallthick_n + 0.05*ecc_n + 0.03*lvarea_n + 0.10*strain_n +
        0.10*wallmotion_n + 0.10*dys_n + 0.10*struct_n
    )
    StructuralScore = clip(StructuralScore, 0, 1)

    # STEP 5: GLOBAL RISK SCORE
    GlobalRisk = (
        0.28*ClinicalScore + 0.24*ElectricalScore + 0.28*StructuralScore +
        0.07*fusion_clinical_ecg + 0.05*fusion_ecg_mri + 0.08*fusion_severity_index
    )
    GlobalRisk = clip(GlobalRisk, 0, 1)

    # STEP 6: PROGRESSION SCORE
    ProgressionScore = (
        0.16*oldpeak_n + 0.10*thalach_n + 0.10*arr_n + 0.10*abn_n +
        0.14*lvef_n + 0.10*wallmotion_n + 0.10*dys_n + 0.10*struct_n +
        0.05*fusion_ecg_mri + 0.05*fusion_severity_index
    )
    ProgressionScore = clip(ProgressionScore, 0, 1)

    # STEP 7: RISK CATEGORY
    if GlobalRisk < 0.40:
        RiskCategory = "Low"
        risk_idx = 0
    elif GlobalRisk < 0.65:
        RiskCategory = "Moderate"
        risk_idx = 1
    else:
        RiskCategory = "High"
        risk_idx = 2

    # STEP 8: RISK PROBABILITIES
    L_low = -4 * GlobalRisk
    L_moderate = 2 - 8 * abs(GlobalRisk - 0.55)
    L_high = 4 * GlobalRisk - 2

    exp_l = math.exp(L_low)
    exp_m = math.exp(L_moderate)
    exp_h = math.exp(L_high)
    denom = exp_l + exp_m + exp_h

    P_low = exp_l / denom
    P_moderate = exp_m / denom
    P_high = exp_h / denom
    risk_probs = [P_low, P_moderate, P_high]

    # STEP 9: MODALITY CONTRIBUTION
    C_clinical = 0.28*ClinicalScore + 0.07*fusion_clinical_ecg
    C_electrical = 0.24*ElectricalScore + 0.04*fusion_clinical_ecg + 0.05*fusion_ecg_mri
    C_structural = 0.28*StructuralScore + 0.03*fusion_ecg_mri + 0.08*fusion_severity_index

    ContributionSum = C_clinical + C_electrical + C_structural
    if ContributionSum == 0: ContributionSum = 1e-5

    clinical_contribution = C_clinical / ContributionSum
    electrical_contribution = C_electrical / ContributionSum
    structural_contribution = C_structural / ContributionSum

    # STEP 10: DISEASE PROGRESS INDEX
    DiseaseProgressIndex = clip(0.6*GlobalRisk + 0.4*ProgressionScore, 0, 1)

    # STEP 11: DISEASE STAGE
    if DiseaseProgressIndex <= 0.25:
        disease_stage = "Minimal burden"
    elif DiseaseProgressIndex <= 0.50:
        disease_stage = "Early disease"
    elif DiseaseProgressIndex <= 0.70:
        disease_stage = "Established disease"
    elif DiseaseProgressIndex <= 0.85:
        disease_stage = "Advanced disease"
    else:
        disease_stage = "Critical progression"

    # STEP 12: DOCTOR-STYLE SUMMARY TEXT
    dominant_score = max(ClinicalScore, ElectricalScore, StructuralScore)
    if dominant_score == ClinicalScore: dominant = "clinical burden"
    elif dominant_score == ElectricalScore: dominant = "electrical instability"
    else: dominant = "structural dysfunction"

    prog_sev = "low" if ProgressionScore < 0.35 else ("moderate" if ProgressionScore < 0.65 else "high")

    if RiskCategory == "High":
        if dominant == "structural dysfunction":
            doctor_summary_text = "Predicted high risk is driven primarily by structural dysfunction, with additional contribution from clinical and electrical burden. Progression tendency is elevated, indicating worsening cardiac status. Disease stage is advanced. Prompt cardiology review and close follow-up are advised."
        else:
            doctor_summary_text = f"Predicted high risk is mainly influenced by marked {dominant}, with significant involvement from other modalities. Progression tendency is {prog_sev}, suggesting increased risk of deterioration. Disease stage is advanced. Urgent specialist assessment is advised."
    elif RiskCategory == "Moderate":
        doctor_summary_text = f"Predicted moderate risk is driven mainly by elevated {dominant}, with moderate electrical and structural involvement. Progression tendency is {prog_sev}. Disease stage suggests established disease. Follow-up cardiac evaluation is recommended."
    else:
        doctor_summary_text = "Predicted low risk is supported by relatively low multimodal burden and limited progression tendency. Disease stage remains early. Routine monitoring and risk-factor management are advised."

    stage_line = {
        "Minimal burden": "Current findings suggest minimal overall burden.",
        "Early disease": "Current findings suggest early-stage cardiac involvement.",
        "Established disease": "Current findings suggest established cardiac disease burden.",
        "Advanced disease": "Current findings suggest advanced cardiac disease.",
        "Critical progression": "Current findings suggest critical progression and high clinical concern."
    }[disease_stage]

    if baseline:
        delta_prog = ProgressionScore - baseline["progression_score"]
        if abs(delta_prog) > 0.05:
            doctor_summary_text += f" Simulated changes {'increased' if delta_prog > 0 else 'decreased'} progression tendency by {abs(delta_prog):.2f}."

    doctor_summary_text += f" {stage_line}"

    # STEP 14: DEBUG LOGGING
    print("----- PROTOTYPE INFERENCE DEBUG -----")
    print(f"Norm Clinical: age={age_n:.2f}, cp={cp_n:.2f}, trestbps={trestbps_n:.2f}, chol={chol_n:.2f}, fbs={fbs_n:.2f}, ecg={restecg_n:.2f}, hr={thalach_n:.2f}, ex={exang_n:.2f}, old={oldpeak_n:.2f}, slope={slope_n:.2f}, ca={ca_n:.2f}, thal={thal_n:.2f}, grp={group_n:.2f}, sex={sex_n:.2f}")
    print(f"Norm Electrical: std={stdrr_n:.2f}, mean={meanrr_n:.2f}, rmssd={rmssd_n:.2f}, pnn={pnn50_n:.2f}, ratio={ratio_n:.2f}, qrs={qrs_n:.2f}, st={st_n:.2f}, rhy={rhythm_n:.2f}, arr={arr_n:.2f}, abn={abn_n:.2f}, hf={hf_n:.2f}")
    print(f"Norm Structural: lvedv={lvedv_n:.2f}, lvesv={lvesv_n:.2f}, rvedv={rvedv_n:.2f}, rvesv={rvesv_n:.2f}, lvef={lvef_n:.2f}, rvef={rvef_n:.2f}, lvsv={lvsv_n:.2f}, mass={lvmass_n:.2f}, thick={wallthick_n:.2f}, ecc={ecc_n:.2f}, area={lvarea_n:.2f}, strain={strain_n:.2f}, motion={wallmotion_n:.2f}, dys={dys_n:.2f}, struct={struct_n:.2f}")
    print(f"Scores -> Clinical: {ClinicalScore:.4f}, Electrical: {ElectricalScore:.4f}, Structural: {StructuralScore:.4f}")
    print(f"GlobalRisk: {GlobalRisk:.4f}, Progression: {ProgressionScore:.4f}, DPI: {DiseaseProgressIndex:.4f}")
    print(f"Probs -> Low: {P_low:.4f}, Mod: {P_moderate:.4f}, High: {P_high:.4f}")
    print(f"Contributions -> {clinical_contribution:.3f}, {electrical_contribution:.3f}, {structural_contribution:.3f}")
    print(f"Stage -> {disease_stage}")
    print("-------------------------------------")

    # STEP 13: RETURN FORMAT
    return {
        "risk_category":     RiskCategory,
        "risk_index":        risk_idx,
        "risk_score":        GlobalRisk,
        "risk_probabilities": {RISK_LABELS[i]: risk_probs[i] for i in range(3)},
        "progression_score": ProgressionScore,
        "disease_progress_index": DiseaseProgressIndex,
        "clinical_score":    ClinicalScore,
        "electrical_score":  ElectricalScore,
        "structural_score":  StructuralScore,
        "modality_attention": {
            "clinical":    clinical_contribution,
            "electrical":  electrical_contribution,
            "structural":  structural_contribution,
        },
        "twin_state":    [round(v, 4) for v in twin_state[:16]],
        "confidence":    max(P_low, P_moderate, P_high),
        "disease_stage": disease_stage,
        "doctor_summary_text": doctor_summary_text,
        "explanation": "Custom evaluation using prototype formulas.",
        "attention_hint": f"{dominant.capitalize()} heavily influences the risk."
    }

# ─────────────────────────────────────────────
# Sample Patients
# ─────────────────────────────────────────────
SAMPLE_PATIENTS = {
    "low": PatientInput(
        clinical=ClinicalFeatures(age=45, sex=0, cp=0, trestbps=120, chol=185, fbs=0, restecg=0, thalach=170, exang=0, oldpeak=0.2, slope=0, ca=0, thal=2, clinical_risk_group=0),
        ecg=ECGFeatures(std_rr=0.04, mean_rr=0.90, rmssd=38.0, pnn50=22.0, low_freq=0.28, high_freq=0.32, spectral_ratio=0.88, p_wave_duration=85.0, qrs_duration=95.0, t_wave_amplitude=0.30, st_segment_elevation=0.0, rhythm_stability_score=0.95, arrhythmia_burden=0.01, abnormality_score=0.05),
        mri=MRIFeatures(lvedv=120.0, lvesv=45.0, rvedv=130.0, rvesv=52.0, lvef=63.0, rvef=60.0, lvsv=75.0, lv_mass=140.0, lv_wall_thickness=8.5, heart_eccentricity=0.45, lv_area=24.0, myocardial_strain=-20.0, wall_motion_score=1.0, dysfunction_score=0.1, structural_abnormality=0.05),
    ),
    "moderate": PatientInput(
        clinical=ClinicalFeatures(age=58, sex=1, cp=1, trestbps=138, chol=240, fbs=1, restecg=1, thalach=145, exang=0, oldpeak=1.5, slope=1, ca=1, thal=2, clinical_risk_group=1),
        ecg=ECGFeatures(std_rr=0.07, mean_rr=0.82, rmssd=25.0, pnn50=10.0, low_freq=0.38, high_freq=0.22, spectral_ratio=1.72, p_wave_duration=95.0, qrs_duration=108.0, t_wave_amplitude=0.20, st_segment_elevation=0.8, rhythm_stability_score=0.6, arrhythmia_burden=0.25, abnormality_score=0.4),
        mri=MRIFeatures(lvedv=155.0, lvesv=65.0, rvedv=158.0, rvesv=68.0, lvef=58.0, rvef=54.0, lvsv=90.0, lv_mass=185.0, lv_wall_thickness=11.0, heart_eccentricity=0.55, lv_area=31.0, myocardial_strain=-16.0, wall_motion_score=1.5, dysfunction_score=0.35, structural_abnormality=0.25),
    ),
    "high": PatientInput(
        clinical=ClinicalFeatures(age=68, sex=1, cp=3, trestbps=160, chol=310, fbs=1, restecg=2, thalach=120, exang=1, oldpeak=3.5, slope=2, ca=3, thal=3, clinical_risk_group=2),
        ecg=ECGFeatures(std_rr=0.11, mean_rr=0.75, rmssd=15.0, pnn50=4.0, low_freq=0.55, high_freq=0.14, spectral_ratio=3.92, p_wave_duration=110.0, qrs_duration=128.0, t_wave_amplitude=0.10, st_segment_elevation=2.5, rhythm_stability_score=0.2, arrhythmia_burden=0.75, abnormality_score=0.85),
        mri=MRIFeatures(lvedv=195.0, lvesv=95.0, rvedv=185.0, rvesv=90.0, lvef=38.0, rvef=35.0, lvsv=100.0, lv_mass=235.0, lv_wall_thickness=13.5, heart_eccentricity=0.70, lv_area=42.0, myocardial_strain=-10.0, wall_motion_score=2.5, dysfunction_score=0.75, structural_abnormality=0.80),
    ),
}

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "Cardiac-FM", "version": "1.0.0"}

@app.get("/sample-patient/{risk_level}")
def get_sample_patient(risk_level: str):
    key = risk_level.lower()
    if key not in SAMPLE_PATIENTS:
        raise HTTPException(status_code=404, detail=f"Unknown risk level '{risk_level}'. Use low/moderate/high.")
    return SAMPLE_PATIENTS[key].dict()

@app.post("/predict")
def predict(patient: PatientInput):
    try:
        # Pass the unscaled Patient pydantic objects to _infer
        result = _infer(patient.clinical, patient.ecg, patient.mri)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
def simulate(req: SimulateRequest):
    try:
        # Baseline
        patient = req.patient
        baseline = _infer(patient.clinical, patient.ecg, patient.mri)

        # Apply deltas
        d = req.deltas
        sim_clinical = copy.deepcopy(patient.clinical)
        sim_ecg      = copy.deepcopy(patient.ecg)
        sim_mri      = copy.deepcopy(patient.mri)

        sim_clinical.chol    += d.delta_chol
        sim_clinical.thalach += d.delta_thalach
        sim_clinical.oldpeak += d.delta_oldpeak
        sim_ecg.std_rr       += d.delta_std_rr
        sim_ecg.rmssd        += d.delta_rmssd
        sim_mri.lvef         += d.delta_lvef
        sim_mri.wall_motion_score     += d.delta_wall_motion_score
        sim_mri.structural_abnormality += d.delta_structural_abnormality

        simulated = _infer(sim_clinical, sim_ecg, sim_mri, baseline=baseline)

        # Multi-step trend (baseline → simulated, linearly interpolated)
        trend = []
        n = req.n_steps
        for i in range(n + 1):
            frac = i / n
            # Interpolate deltas
            step_c = copy.deepcopy(patient.clinical)
            step_e = copy.deepcopy(patient.ecg)
            step_m = copy.deepcopy(patient.mri)
            step_c.chol    += d.delta_chol * frac
            step_c.thalach += d.delta_thalach * frac
            step_c.oldpeak += d.delta_oldpeak * frac
            step_e.std_rr  += d.delta_std_rr * frac
            step_e.rmssd   += d.delta_rmssd * frac
            step_m.lvef    += d.delta_lvef * frac
            step_m.wall_motion_score      += d.delta_wall_motion_score * frac
            step_m.structural_abnormality += d.delta_structural_abnormality * frac

            r = _infer(step_c, step_e, step_m)
            trend.append({
                "step": i,
                "progression_score": r["progression_score"],
                "risk_score":        r["risk_score"],
                "risk_category":     r["risk_category"],
            })

        return {
            "baseline":  baseline,
            "simulated": simulated,
            "trend":     trend,
            "delta_progression": round(simulated["progression_score"] - baseline["progression_score"], 4),
            "delta_risk_score":  round(simulated["risk_score"] - baseline["risk_score"], 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
