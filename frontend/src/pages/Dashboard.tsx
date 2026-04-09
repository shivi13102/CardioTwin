// src/pages/Dashboard.tsx — Landing page with quick actions

import { motion } from 'framer-motion';
import { Heart, Zap, BarChart2, Activity, ChevronRight, Cpu } from 'lucide-react';
import type { Page, PatientInput } from '../types';
import { getSamplePatient } from '../api';
import { useState } from 'react';

interface Props {
  onLoadPatient: (p: PatientInput) => void;
  onNavigate: (p: Page) => void;
}

const features = [
  {
    icon: <Heart size={20} className="text-rose-400" />,
    title: 'Multimodal Fusion',
    desc: 'Integrates EHR, ECG, and MRI into a unified cardiac latent state.',
    bg: 'bg-rose-500/10 border-rose-500/20',
  },
  {
    icon: <Cpu size={20} className="text-primary-400" />,
    title: 'Cardiac-FM Model',
    desc: 'Foundation model with modality-specific encoders and a shared twin state.',
    bg: 'bg-primary-500/10 border-primary-500/20',
  },
  {
    icon: <Zap size={20} className="text-amber-400" />,
    title: 'What-If Simulation',
    desc: 'Perturb patient features and observe risk trajectory changes in real time.',
    bg: 'bg-amber-500/10 border-amber-500/20',
  },
  {
    icon: <BarChart2 size={20} className="text-violet-400" />,
    title: 'Risk Stratification',
    desc: 'Predicts Low / Moderate / High cardiac risk with interpretable confidence.',
    bg: 'bg-violet-500/10 border-violet-500/20',
  },
];

export default function Dashboard({ onLoadPatient, onNavigate }: Props) {
  const [loading, setLoading] = useState<string | null>(null);

  const loadSample = async (level: 'low' | 'moderate' | 'high') => {
    setLoading(level);
    try {
      const patient = await getSamplePatient(level);
      onLoadPatient(patient);
    } catch {
      alert('Could not connect to backend. Please start the API server first.');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-full p-8 max-w-5xl mx-auto">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-12"
      >
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xs font-semibold px-3 py-1 bg-primary-600/20 text-primary-400 border border-primary-600/30 rounded-full tracking-widest uppercase">
            Research Prototype
          </span>
          <span className="text-xs text-slate-500 flex items-center gap-1">
            <Activity size={12} /> Cardiac-FM v1.0
          </span>
        </div>

        <h1 className="text-5xl font-extrabold text-white mb-4 leading-tight">
          <span className="gradient-text">CardioTwin</span>
        </h1>
        <p className="text-xl text-slate-300 font-medium mb-2">
          Multimodal Cardiac Digital Twin Framework
        </p>
        <p className="text-slate-500 max-w-2xl leading-relaxed">
          A research-grade AI system that fuses EHR, ECG, and MRI data to create a personalized
          cardiac digital twin — predicting risk trajectory and enabling what-if disease simulation
          for clinical decision support.
        </p>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.4 }}
        className="mb-10"
      >
        <p className="section-title">Quick Actions</p>
        <div className="flex flex-wrap gap-3">
          <button
            className="btn-primary"
            onClick={() => onNavigate('patient')}
          >
            <Heart size={16} />
            Enter Patient Data
            <ChevronRight size={14} />
          </button>
          <button
            className="btn-secondary"
            onClick={() => onNavigate('simulation')}
          >
            <Zap size={16} />
            Open Simulation Panel
          </button>
        </div>
      </motion.div>

      {/* Sample Presets */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.4 }}
        className="mb-10"
      >
        <p className="section-title">Load Sample Patient</p>
        <div className="grid grid-cols-3 gap-4">
          {(['low', 'moderate', 'high'] as const).map((level, idx) => {
            const styles = {
              low:      { border: 'border-emerald-500/30 hover:border-emerald-500/60', dot: 'bg-emerald-400', text: 'text-emerald-400', label: 'Low Risk Patient' },
              moderate: { border: 'border-amber-500/30 hover:border-amber-500/60',    dot: 'bg-amber-400',   text: 'text-amber-400',   label: 'Moderate Risk Patient' },
              high:     { border: 'border-rose-500/30 hover:border-rose-500/60',      dot: 'bg-rose-400',    text: 'text-rose-400',    label: 'High Risk Patient' },
            }[level];

            return (
              <motion.button
                key={level}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => loadSample(level)}
                disabled={loading !== null}
                className={`card ${styles.border} p-5 text-left transition-all duration-200 disabled:opacity-50 cursor-pointer`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className={`w-2.5 h-2.5 rounded-full ${styles.dot}`} />
                  <span className={`text-xs font-bold uppercase tracking-wider ${styles.text}`}>{level}</span>
                </div>
                <p className="font-semibold text-white text-sm mb-1">{styles.label}</p>
                <p className="text-xs text-slate-500">
                  {loading === level ? 'Loading...' : 'Click to load preset patient data →'}
                </p>
              </motion.button>
            );
          })}
        </div>
      </motion.div>

      {/* Features */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.4 }}
      >
        <p className="section-title">System Capabilities</p>
        <div className="grid grid-cols-2 gap-4">
          {features.map((f, i) => (
            <div key={i} className={`card border ${f.bg} p-5 flex gap-4`}>
              <div className="mt-0.5">{f.icon}</div>
              <div>
                <p className="font-semibold text-white text-sm mb-1">{f.title}</p>
                <p className="text-xs text-slate-500 leading-relaxed">{f.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Architecture note */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 card border border-slate-800 p-5"
      >
        <p className="section-title">Model Architecture</p>
        <div className="flex items-center gap-2 text-sm text-slate-400 flex-wrap">
          {['EHR Encoder (MLP)', '→', 'ECG Encoder (MLP)', '→', 'MRI Encoder (MLP)', '→',
            'Fusion MLP + Gating', '→', 'Twin State', '→', 'Risk Head & Progression Head'].map((s, i) => (
            <span key={i} className={s === '→' ? 'text-slate-600' : 'px-2 py-1 bg-slate-800 rounded text-xs font-mono'}>{s}</span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
