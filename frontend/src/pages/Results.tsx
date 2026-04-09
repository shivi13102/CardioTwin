// src/pages/Results.tsx — Digital Twin Analysis results dashboard

import { motion } from 'framer-motion';
import {
  Heart, Activity, Scan, BrainCircuit, ChevronRight, ArrowLeft,
  TrendingUp, ShieldAlert, Zap, Info
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import type { PredictResult, PatientInput } from '../types';

interface Props {
  result: PredictResult | null;
  patient: PatientInput | null;
  onSimulate: () => void;
  onBack: () => void;
}

const RISK_STYLES = {
  Low:      { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', bar: '#10b981' },
  Moderate: { bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   text: 'text-amber-400',   bar: '#f59e0b' },
  High:     { bg: 'bg-rose-500/10',    border: 'border-rose-500/30',    text: 'text-rose-400',    bar: '#f43f5e' },
};

function ScoreCard({ title, value, subtitle, icon, color }: { title: string; value: string | number; subtitle: string; icon: React.ReactNode; color: string }) {
  return (
    <div className={`card p-5 border ${color}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="text-slate-400">{icon}</div>
        <span className="text-xs text-slate-600 font-medium">{subtitle}</span>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-slate-500">{title}</div>
    </div>
  );
}

function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function ResultsPage({ result, patient, onSimulate, onBack }: Props) {
  if (!result) {
    return (
      <div className="min-h-full flex items-center justify-center p-8">
        <div className="text-center">
          <BrainCircuit size={48} className="text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400 text-lg font-medium">No results yet</p>
          <p className="text-slate-600 text-sm mt-1">Go to Patient Input and generate a digital twin first.</p>
          <button onClick={onBack} className="btn-secondary mt-4">
            <ArrowLeft size={14} /> Go to Patient Input
          </button>
        </div>
      </div>
    );
  }

  const riskStyle = RISK_STYLES[result.risk_category] ?? RISK_STYLES.Moderate;

  const radarData = [
    { subject: 'Clinical',    value: Math.round(result.clinical_score * 100) },
    { subject: 'Electrical',  value: Math.round(result.electrical_score * 100) },
    { subject: 'Structural',  value: Math.round(result.structural_score * 100) },
    { subject: 'Progression', value: Math.round(result.progression_score * 100) },
    { subject: 'Risk',        value: Math.round(result.risk_score * 100) },
  ];

  const probData = Object.entries(result.risk_probabilities).map(([k, v]) => ({
    name: k, value: parseFloat((v * 100).toFixed(1)), fill: RISK_STYLES[k as 'Low' | 'Moderate' | 'High']?.bar ?? '#64748b',
  }));

  const attentionData = [
    { name: 'Clinical',   value: parseFloat((result.modality_attention.clinical * 100).toFixed(1)),   fill: '#3b82f6' },
    { name: 'Electrical', value: parseFloat((result.modality_attention.electrical * 100).toFixed(1)), fill: '#f59e0b' },
    { name: 'Structural', value: parseFloat((result.modality_attention.structural * 100).toFixed(1)), fill: '#a78bfa' },
  ];

  return (
    <div className="min-h-full p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <button onClick={onBack} className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 mb-2 transition-colors">
            <ArrowLeft size={12} /> Back to Patient Input
          </button>
          <h1 className="text-3xl font-bold text-white">Cardiac Digital Twin</h1>
          <p className="text-slate-500 text-sm mt-1">Analysis powered by Cardiac-FM · {new Date().toLocaleDateString()}</p>
        </div>
        <button onClick={onSimulate} className="btn-primary">
          <Zap size={16} /> Run Simulation <ChevronRight size={14} />
        </button>
      </div>

      {/* Risk Banner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className={`card border ${riskStyle.border} ${riskStyle.bg} p-6 mb-6 flex items-center gap-6`}
      >
        <div className="relative">
          <div className={`w-16 h-16 rounded-2xl ${riskStyle.bg} border ${riskStyle.border} flex items-center justify-center twin-ring`}>
            <Heart size={28} className={riskStyle.text} fill="currentColor" />
          </div>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <span className={`text-3xl font-extrabold ${riskStyle.text}`}>{result.risk_category} Risk</span>
            <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${riskStyle.bg} ${riskStyle.border} ${riskStyle.text}`}>
              {(result.risk_score * 100).toFixed(1)}% confidence
            </span>
          </div>
          <p className="text-slate-400 text-sm max-w-xl">{result.explanation}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500 mb-1">CONFIDENCE</p>
          <p className="text-2xl font-bold text-white">{(result.confidence * 100).toFixed(0)}%</p>
        </div>
      </motion.div>

      {/* Score Cards Row */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <ScoreCard title="Risk Category" value={result.risk_category} subtitle="Classification" icon={<ShieldAlert size={16} />} color={`border-${result.risk_index === 0 ? 'emerald' : result.risk_index === 1 ? 'amber' : 'rose'}-500/30`} />
        <ScoreCard title="Disease Stage" value={result.disease_stage} subtitle="Staging" icon={<Activity size={16} />} color="border-slate-700" />
        <ScoreCard title="Global Risk Score" value={(result.risk_score * 100).toFixed(1) + '%'} subtitle="Fused" icon={<BrainCircuit size={16} />} color="border-primary-500/30" />
        <ScoreCard title="Disease Progress Index" value={(result.disease_progress_index * 100).toFixed(1) + '%'} subtitle="DPI" icon={<TrendingUp size={16} />} color="border-slate-700" />
        
        <ScoreCard title="Progression Score" value={(result.progression_score * 100).toFixed(1) + '%'} subtitle="Trajectory" icon={<TrendingUp size={16} />} color="border-slate-700" />
        <ScoreCard title="Clinical Score" value={(result.clinical_score * 100).toFixed(1) + '%'} subtitle="EHR Branch" icon={<Heart size={16} />} color="border-slate-700" />
        <ScoreCard title="Electrical Score" value={(result.electrical_score * 100).toFixed(1) + '%'} subtitle="ECG Branch" icon={<Activity size={16} />} color="border-slate-700" />
        <ScoreCard title="Structural Score" value={(result.structural_score * 100).toFixed(1) + '%'} subtitle="MRI Branch" icon={<Scan size={16} />} color="border-slate-700" />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* Radar */}
        <div className="card border border-slate-800 p-5">
          <p className="section-title mb-4">Twin Profile</p>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} />
              <Radar name="Patient" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Probabilities */}
        <div className="card border border-slate-800 p-5">
          <p className="section-title mb-4">Risk Probabilities</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={probData} layout="vertical">
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {probData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Modality Attention */}
        <div className="card border border-slate-800 p-5">
          <p className="section-title mb-4">Modality Attention</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={attentionData} layout="vertical">
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} width={70} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {attentionData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Modality Scores */}
      <div className="card border border-slate-800 p-6 mb-6">
        <p className="section-title mb-4">Modality Contribution Breakdown</p>
        <div className="space-y-3">
          <ProgressBar label="Clinical Burden (EHR)" value={result.clinical_score} color="#3b82f6" />
          <ProgressBar label="Electrical Instability (ECG)" value={result.electrical_score} color="#f59e0b" />
          <ProgressBar label="Structural Dysfunction (MRI)" value={result.structural_score} color="#a78bfa" />
          <ProgressBar label="Progression Tendency" value={result.progression_score} color="#f43f5e" />
        </div>
      </div>

      {/* Clinical Insight + Twin State */}
      <div className="grid grid-cols-2 gap-4">
        <div className="card border border-slate-800 p-5">
          <div className="flex items-center gap-2 mb-3">
            <Info size={14} className="text-primary-400" />
            <p className="section-title mb-0">Doctor's Summary</p>
          </div>
          <p className="text-slate-300 text-sm leading-relaxed">{result.doctor_summary_text}</p>
        </div>

        <div className="card border border-slate-800 p-5">
          <p className="section-title mb-3">Twin State Preview (first 16 dims)</p>
          <div className="grid grid-cols-8 gap-1">
            {result.twin_state.slice(0, 16).map((v, i) => {
              // Normalize the raw latent value strictly for visual bar height to prevent overflow
              const normalizedHeight = Math.min(Math.abs(v) * 20, 100); 
              const colorAlpha = Math.min(Math.abs(v) * 0.3, 0.8);
              return (
                <div key={i} className="text-center">
                  <div
                    className="h-8 rounded flex items-end justify-center overflow-hidden mb-1"
                    style={{ backgroundColor: `rgba(59,130,246,${colorAlpha})` }}
                  >
                    <div className="w-full" style={{ height: `${normalizedHeight}%`, backgroundColor: '#3b82f6', borderRadius: '2px' }} />
                  </div>
                  <span className="text-[9px] text-slate-500 font-mono">z{i}</span>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-slate-600 mt-2">Fused latent representation</p>
        </div>
      </div>
    </div>
  );
}
