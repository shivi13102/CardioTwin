// src/types.ts — Shared type definitions for CardioTwin

export interface ClinicalFeatures {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
  clinical_risk_group: number;
}

export interface ECGFeatures {
  std_rr: number;
  mean_rr: number;
  rmssd: number;
  pnn50: number;
  low_freq: number;
  high_freq: number;
  spectral_ratio: number;
  p_wave_duration: number;
  qrs_duration: number;
  t_wave_amplitude: number;
  st_segment_elevation: number;
  rhythm_stability_score: number;
  arrhythmia_burden: number;
  abnormality_score: number;
}

export interface MRIFeatures {
  lvedv: number;
  lvesv: number;
  rvedv: number;
  rvesv: number;
  lvef: number;
  rvef: number;
  lvsv: number;
  lv_mass: number;
  lv_wall_thickness: number;
  heart_eccentricity: number;
  lv_area: number;
  myocardial_strain: number;
  wall_motion_score: number;
  dysfunction_score: number;
  structural_abnormality: number;
}

export interface PatientInput {
  clinical: ClinicalFeatures;
  ecg: ECGFeatures;
  mri: MRIFeatures;
}

export interface ModalityAttention {
  clinical: number;
  electrical: number;
  structural: number;
}

export interface PredictResult {
  risk_category: 'Low' | 'Moderate' | 'High';
  risk_index: number;
  risk_score: number;
  risk_probabilities: Record<string, number>;
  progression_score: number;
  clinical_score: number;
  electrical_score: number;
  structural_score: number;
  modality_attention: ModalityAttention;
  twin_state: number[];
  confidence: number;
  explanation: string;
  attention_hint: string;
  disease_progress_index: number;
  disease_stage: string;
  doctor_summary_text: string;
}

export interface SimulationDeltas {
  delta_chol: number;
  delta_thalach: number;
  delta_oldpeak: number;
  delta_lvef: number;
  delta_wall_motion_score: number;
  delta_structural_abnormality: number;
  delta_std_rr: number;
  delta_rmssd: number;
}

export interface TrendPoint {
  step: number;
  progression_score: number;
  risk_score: number;
  risk_category: string;
}

export interface SimulateResult {
  baseline: PredictResult;
  simulated: PredictResult;
  trend: TrendPoint[];
  delta_progression: number;
  delta_risk_score: number;
}

export type Page = 'dashboard' | 'patient' | 'results' | 'simulation' | 'datainfo';
export type RiskLevel = 'low' | 'moderate' | 'high';
