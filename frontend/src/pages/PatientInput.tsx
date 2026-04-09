// src/pages/PatientInput.tsx — Tabbed patient data entry with presets and validation

import { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Activity, Scan, BrainCircuit, ChevronDown, ChevronRight, Loader2 } from 'lucide-react';
import type { PatientInput, ClinicalFeatures, ECGFeatures, MRIFeatures, PredictResult } from '../types';
import { predictTwin, getSamplePatient } from '../api';

interface Props {
  initialPatient: PatientInput | null;
  onResultReady: (p: PatientInput, r: PredictResult) => void;
}

type TabId = 'clinical' | 'ecg' | 'mri';

const defaultPatient: PatientInput = {
  clinical: { age: 55, sex: 1, cp: 0, trestbps: 130, chol: 230, fbs: 0, restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0, slope: 1, ca: 0, thal: 2, clinical_risk_group: 0 },
  ecg: { std_rr: 0.05, mean_rr: 0.85, rmssd: 30.0, pnn50: 15.0, low_freq: 0.3, high_freq: 0.25, spectral_ratio: 1.2, p_wave_duration: 90, qrs_duration: 100, t_wave_amplitude: 0.25, st_segment_elevation: 0.1, rhythm_stability_score: 0.85, arrhythmia_burden: 0.05, abnormality_score: 0.1 },
  mri: { lvedv: 140, lvesv: 55, rvedv: 145, rvesv: 60, lvef: 60, rvef: 58, lvsv: 85, lv_mass: 160, lv_wall_thickness: 9.5, heart_eccentricity: 0.5, lv_area: 28, myocardial_strain: -18, wall_motion_score: 1.0, dysfunction_score: 0.2, structural_abnormality: 0.1 },
};

function FieldRow({ label, unit, value, onChange, min, max, step }: { label: string; unit?: string; value: number | ''; onChange: (v: number | '') => void; min?: number; max?: number; step?: number }) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-xs text-slate-400 w-44 shrink-0">{label}{unit && <span className="text-slate-600 ml-1">({unit})</span>}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step ?? 0.01}
        onChange={e => onChange(e.target.value === '' ? '' : parseFloat(e.target.value))}
        className="input-field flex-1 text-right font-mono"
      />
    </div>
  );
}

export default function PatientInputPage({ initialPatient, onResultReady }: Props) {
  const [patient, setPatient] = useState<PatientInput>(initialPatient ?? defaultPatient);
  const [activeTab, setActiveTab] = useState<TabId>('clinical');
  const [loading, setLoading] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState<string | null>(null);
  const [error, setError] = useState('');

  const updateClinical = (key: keyof ClinicalFeatures, val: number | '') =>
    setPatient(p => ({ ...p, clinical: { ...p.clinical, [key]: val as any } }));

  const updateECG = (key: keyof ECGFeatures, val: number | '') =>
    setPatient(p => ({ ...p, ecg: { ...p.ecg, [key]: val as any } }));

  const updateMRI = (key: keyof MRIFeatures, val: number | '') =>
    setPatient(p => ({ ...p, mri: { ...p.mri, [key]: val as any } }));

  const loadPreset = async (level: string) => {
    setLoadingPreset(level);
    try {
      const p = await getSamplePatient(level);
      setPatient(p);
    } catch {
      setError('Backend offline. Please run: uvicorn api.main:app --reload');
    } finally {
      setLoadingPreset(null);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    
    // Replace any empty strings with 0 so the backend doesn't crash on validation
    const sanitize = (obj: any) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, v === '' ? 0 : v]));
    const cleanPatient = {
      clinical: sanitize(patient.clinical),
      ecg: sanitize(patient.ecg),
      mri: sanitize(patient.mri),
    } as PatientInput;

    try {
      const result = await predictTwin(cleanPatient);
      onResultReady(cleanPatient, result);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? 'Could not connect to backend. Please ensure the API is running at http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'clinical', label: 'Clinical (EHR)', icon: <User size={15} /> },
    { id: 'ecg',      label: 'Electrical (ECG)', icon: <Activity size={15} /> },
    { id: 'mri',      label: 'Structural (MRI)', icon: <Scan size={15} /> },
  ];

  return (
    <div className="min-h-full p-8 max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">Patient Data Entry</h1>
        <p className="text-slate-500 text-sm">Enter multimodal patient features across all three modalities, then generate the digital twin.</p>
      </div>

      {/* Presets */}
      <div className="mb-6">
        <p className="section-title">Presets</p>
        <div className="flex gap-2">
          {['low', 'moderate', 'high'].map(level => (
            <button key={level} onClick={() => loadPreset(level)} disabled={loadingPreset !== null}
              className="btn-secondary text-xs py-1.5 px-3 capitalize disabled:opacity-50">
              {loadingPreset === level ? <Loader2 size={12} className="animate-spin" /> : null}
              {level} Risk
            </button>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 mb-6 gap-1">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all duration-200 border-b-2 -mb-px ${
              activeTab === t.id
                ? 'border-primary-500 text-primary-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab Panels */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        className="card p-6 space-y-3 mb-6"
      >
        {activeTab === 'clinical' && (
          <>
            <p className="section-title">Clinical / EHR Features</p>
            <FieldRow label="Age" unit="years" value={patient.clinical.age} onChange={v => updateClinical('age', v)} min={18} max={100} step={1} />
            <FieldRow label="Sex" unit="0=F,1=M" value={patient.clinical.sex} onChange={v => updateClinical('sex', v)} min={0} max={1} step={1} />
            <FieldRow label="Chest Pain Type (cp)" unit="0-3" value={patient.clinical.cp} onChange={v => updateClinical('cp', v)} min={0} max={3} step={1} />
            <FieldRow label="Resting BP (trestbps)" unit="mmHg" value={patient.clinical.trestbps} onChange={v => updateClinical('trestbps', v)} min={80} max={220} step={1} />
            <FieldRow label="Cholesterol (chol)" unit="mg/dL" value={patient.clinical.chol} onChange={v => updateClinical('chol', v)} min={100} max={600} step={1} />
            <FieldRow label="Fasting Blood Sugar (fbs)" unit="0/1" value={patient.clinical.fbs} onChange={v => updateClinical('fbs', v)} min={0} max={1} step={1} />
            <FieldRow label="Resting ECG (restecg)" unit="0-2" value={patient.clinical.restecg} onChange={v => updateClinical('restecg', v)} min={0} max={2} step={1} />
            <FieldRow label="Max Heart Rate (thalach)" unit="bpm" value={patient.clinical.thalach} onChange={v => updateClinical('thalach', v)} min={60} max={250} step={1} />
            <FieldRow label="Exercise Angina (exang)" unit="0/1" value={patient.clinical.exang} onChange={v => updateClinical('exang', v)} min={0} max={1} step={1} />
            <FieldRow label="ST Depression (oldpeak)" unit="mm" value={patient.clinical.oldpeak} onChange={v => updateClinical('oldpeak', v)} min={0} max={10} />
            <FieldRow label="Slope" unit="0-2" value={patient.clinical.slope} onChange={v => updateClinical('slope', v)} min={0} max={2} step={1} />
            <FieldRow label="Major Vessels (ca)" unit="0-3" value={patient.clinical.ca} onChange={v => updateClinical('ca', v)} min={0} max={4} step={1} />
            <FieldRow label="Thalassemia (thal)" unit="1-3" value={patient.clinical.thal} onChange={v => updateClinical('thal', v)} min={0} max={3} step={1} />
            <FieldRow label="Risk Group" unit="0=Low,1=Mod,2=High" value={patient.clinical.clinical_risk_group} onChange={v => updateClinical('clinical_risk_group', v)} min={0} max={2} step={1} />
          </>
        )}

        {activeTab === 'ecg' && (
          <>
            <p className="section-title">Electrical / ECG Features (HRV)</p>
            <FieldRow label="Std RR Interval (std_rr)" unit="s" value={patient.ecg.std_rr} onChange={v => updateECG('std_rr', v)} />
            <FieldRow label="Mean RR Interval" unit="s" value={patient.ecg.mean_rr} onChange={v => updateECG('mean_rr', v)} />
            <FieldRow label="RMSSD" unit="ms" value={patient.ecg.rmssd} onChange={v => updateECG('rmssd', v)} />
            <FieldRow label="pNN50" unit="%" value={patient.ecg.pnn50} onChange={v => updateECG('pnn50', v)} />
            <FieldRow label="Low Frequency Power" unit="nu" value={patient.ecg.low_freq} onChange={v => updateECG('low_freq', v)} />
            <FieldRow label="High Frequency Power" unit="nu" value={patient.ecg.high_freq} onChange={v => updateECG('high_freq', v)} />
            <FieldRow label="LF/HF Spectral Ratio" value={patient.ecg.spectral_ratio} onChange={v => updateECG('spectral_ratio', v)} />
            <FieldRow label="P-Wave Duration" unit="ms" value={patient.ecg.p_wave_duration} onChange={v => updateECG('p_wave_duration', v)} />
            <FieldRow label="QRS Duration" unit="ms" value={patient.ecg.qrs_duration} onChange={v => updateECG('qrs_duration', v)} />
            <FieldRow label="T-Wave Amplitude" unit="mV" value={patient.ecg.t_wave_amplitude} onChange={v => updateECG('t_wave_amplitude', v)} />
            <FieldRow label="ST Segment Elevation" unit="mm" value={patient.ecg.st_segment_elevation} onChange={v => updateECG('st_segment_elevation', v)} />
            <FieldRow label="Rhythm Stability Score" unit="0-1" value={patient.ecg.rhythm_stability_score} onChange={v => updateECG('rhythm_stability_score', v)} min={0} max={1} />
            <FieldRow label="Arrhythmia Burden" unit="0-1" value={patient.ecg.arrhythmia_burden} onChange={v => updateECG('arrhythmia_burden', v)} min={0} max={1} />
            <FieldRow label="Abnormality Score" unit="0-1" value={patient.ecg.abnormality_score} onChange={v => updateECG('abnormality_score', v)} min={0} max={1} />
          </>
        )}

        {activeTab === 'mri' && (
          <>
            <p className="section-title">Structural / Cardiac MRI Features</p>
            <FieldRow label="LV End-Diastolic Vol (LVEDV)" unit="mL" value={patient.mri.lvedv} onChange={v => updateMRI('lvedv', v)} />
            <FieldRow label="LV End-Systolic Vol (LVESV)" unit="mL" value={patient.mri.lvesv} onChange={v => updateMRI('lvesv', v)} />
            <FieldRow label="RV End-Diastolic Vol (RVEDV)" unit="mL" value={patient.mri.rvedv} onChange={v => updateMRI('rvedv', v)} />
            <FieldRow label="RV End-Systolic Vol (RVESV)" unit="mL" value={patient.mri.rvesv} onChange={v => updateMRI('rvesv', v)} />
            <FieldRow label="LV Ejection Fraction (LVEF)" unit="%" value={patient.mri.lvef} onChange={v => updateMRI('lvef', v)} min={10} max={90} />
            <FieldRow label="RV Ejection Fraction (RVEF)" unit="%" value={patient.mri.rvef} onChange={v => updateMRI('rvef', v)} min={10} max={90} />
            <FieldRow label="LV Stroke Volume (LVSV)" unit="mL" value={patient.mri.lvsv} onChange={v => updateMRI('lvsv', v)} />
            <FieldRow label="LV Mass" unit="g" value={patient.mri.lv_mass} onChange={v => updateMRI('lv_mass', v)} />
            <FieldRow label="LV Wall Thickness" unit="mm" value={patient.mri.lv_wall_thickness} onChange={v => updateMRI('lv_wall_thickness', v)} />
            <FieldRow label="Heart Eccentricity" value={patient.mri.heart_eccentricity} onChange={v => updateMRI('heart_eccentricity', v)} />
            <FieldRow label="LV Area" unit="cm²" value={patient.mri.lv_area} onChange={v => updateMRI('lv_area', v)} />
            <FieldRow label="Myocardial Strain" unit="%" value={patient.mri.myocardial_strain} onChange={v => updateMRI('myocardial_strain', v)} />
            <FieldRow label="Wall Motion Score" unit="1-3" value={patient.mri.wall_motion_score} onChange={v => updateMRI('wall_motion_score', v)} min={1} max={4} />
            <FieldRow label="Dysfunction Score" unit="0-1" value={patient.mri.dysfunction_score} onChange={v => updateMRI('dysfunction_score', v)} min={0} max={1} />
            <FieldRow label="Structural Abnormality" unit="0-1" value={patient.mri.structural_abnormality} onChange={v => updateMRI('structural_abnormality', v)} min={0} max={1} />
          </>
        )}
      </motion.div>

      {error && (
        <div className="mb-4 px-4 py-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm">
          {error}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="btn-primary w-full justify-center py-3 text-base disabled:opacity-60"
      >
        {loading ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
        {loading ? 'Generating Digital Twin…' : 'Generate Cardiac Digital Twin'}
        {!loading && <ChevronRight size={16} />}
      </button>
    </div>
  );
}
