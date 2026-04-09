// src/pages/Simulation.tsx — What-If Simulation Panel

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Zap, ArrowLeft, RotateCcw, Loader2, TrendingUp, TrendingDown,
  Minus, Activity, Heart, Scan
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Legend, BarChart, Bar, Cell
} from 'recharts';
import type { PatientInput, PredictResult, SimulateResult, SimulationDeltas } from '../types';
import { simulateTwin } from '../api';

interface Props {
  patient: PatientInput | null;
  baselineResult: PredictResult | null;
  simResult: SimulateResult | null;
  onSimDone: (s: SimulateResult) => void;
  onBack: () => void;
}

const DEFAULT_DELTAS: SimulationDeltas = {
  delta_chol: 0, delta_thalach: 0, delta_oldpeak: 0,
  delta_lvef: 0, delta_wall_motion_score: 0, delta_structural_abnormality: 0,
  delta_std_rr: 0, delta_rmssd: 0,
};

interface SliderConfig {
  key: keyof SimulationDeltas;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  icon: React.ReactNode;
  group: 'clinical' | 'electrical' | 'structural';
  tip: string;
}

const SLIDERS: SliderConfig[] = [
  { key: 'delta_chol', label: 'Δ Cholesterol', unit: 'mg/dL', min: -100, max: 150, step: 5, icon: <Heart size={13} />, group: 'clinical', tip: '+ve = worsening' },
  { key: 'delta_thalach', label: 'Δ Max Heart Rate', unit: 'bpm', min: -60, max: 40, step: 2, icon: <Heart size={13} />, group: 'clinical', tip: '-ve = worsening' },
  { key: 'delta_oldpeak', label: 'Δ ST Depression', unit: 'mm', min: -2, max: 4, step: 0.1, icon: <Heart size={13} />, group: 'clinical', tip: '+ve = worsening' },
  { key: 'delta_std_rr', label: 'Δ RR Std Dev', unit: 's', min: -0.05, max: 0.1, step: 0.005, icon: <Activity size={13} />, group: 'electrical', tip: '+ve = arrhythmia risk' },
  { key: 'delta_rmssd', label: 'Δ RMSSD', unit: 'ms', min: -30, max: 30, step: 1, icon: <Activity size={13} />, group: 'electrical', tip: '-ve = worsening HRV' },
  { key: 'delta_lvef', label: 'Δ LVEF', unit: '%', min: -30, max: 20, step: 1, icon: <Scan size={13} />, group: 'structural', tip: '-ve = worsening' },
  { key: 'delta_wall_motion_score', label: 'Δ Wall Motion Score', unit: '', min: -1, max: 2, step: 0.1, icon: <Scan size={13} />, group: 'structural', tip: '+ve = worsening' },
  { key: 'delta_structural_abnormality', label: 'Δ Structural Abnormality', unit: '0-1', min: -0.5, max: 0.5, step: 0.05, icon: <Scan size={13} />, group: 'structural', tip: '+ve = worsening' },
];

const GROUP_STYLE = {
  clinical:   { label: 'Clinical', color: '#3b82f6', icon: <Heart size={14} className="text-blue-400" /> },
  electrical: { label: 'Electrical', color: '#f59e0b', icon: <Activity size={14} className="text-amber-400" /> },
  structural: { label: 'Structural', color: '#a78bfa', icon: <Scan size={14} className="text-violet-400" /> },
};

const RISK_COLOR = { Low: '#10b981', Moderate: '#f59e0b', High: '#f43f5e' };

function DeltaBadge({ value }: { value: number }) {
  if (Math.abs(value) < 0.001) return <span className="text-slate-500 text-xs font-mono">–</span>;
  const positive = value > 0;
  return (
    <span className={`flex items-center gap-0.5 text-xs font-semibold ${positive ? 'text-rose-400' : 'text-emerald-400'}`}>
      {positive ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
      {positive ? '+' : ''}{value > 0 ? value.toFixed(3) : value.toFixed(3)}
    </span>
  );
}

function CompareRow({ label, baseline, simulated }: { label: string; baseline: number; simulated: number }) {
  const delta = simulated - baseline;
  return (
    <div className="flex items-center gap-3 py-2 border-b border-slate-800 last:border-0">
      <span className="text-xs text-slate-400 flex-1">{label}</span>
      <span className="font-mono text-xs text-slate-300 w-14 text-right">{(baseline * 100).toFixed(1)}%</span>
      <span className="text-slate-600 text-xs">→</span>
      <span className="font-mono text-xs text-white w-14 text-right">{(simulated * 100).toFixed(1)}%</span>
      <div className="w-20 text-right"><DeltaBadge value={parseFloat(delta.toFixed(3))} /></div>
    </div>
  );
}

export default function SimulationPage({ patient, baselineResult, simResult, onSimDone, onBack }: Props) {
  const [deltas, setDeltas] = useState<SimulationDeltas>(DEFAULT_DELTAS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [nSteps, setNSteps] = useState(5);

  const setDelta = useCallback((key: keyof SimulationDeltas, val: number) => {
    setDeltas(d => ({ ...d, [key]: val }));
  }, []);

  const handleSimulate = async () => {
    if (!patient) { setError('No patient loaded. Go to Patient Input first.'); return; }
    setLoading(true);
    setError('');
    try {
      const result = await simulateTwin(patient, deltas, nSteps);
      onSimDone(result);
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      const msg = typeof detail === 'string'
        ? detail
        : (Array.isArray(detail) ? 'Validation Error: Check your inputs' : 'Backend offline. Run: uvicorn api.main:app --reload');
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const resetDeltas = () => setDeltas(DEFAULT_DELTAS);

  const trendData = simResult?.trend.map((t) => ({
    step: `T${t.step}`,
    progression: parseFloat((t.progression_score * 100).toFixed(1)),
    risk: parseFloat((t.risk_score * 100).toFixed(1)),
    category: t.risk_category,
  })) ?? [];

  const compareData = simResult ? [
    { name: 'Clinical',   baseline: simResult.baseline.clinical_score,   simulated: simResult.simulated.clinical_score },
    { name: 'Electrical', baseline: simResult.baseline.electrical_score,  simulated: simResult.simulated.electrical_score },
    { name: 'Structural', baseline: simResult.baseline.structural_score,  simulated: simResult.simulated.structural_score },
  ] : [];

  const grouped = ['clinical', 'electrical', 'structural'] as const;

  return (
    <div className="min-h-full p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <button onClick={onBack} className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 mb-2 transition-colors">
            <ArrowLeft size={12} /> Back to Results
          </button>
          <h1 className="text-3xl font-bold text-white">What-If Simulation</h1>
          <p className="text-slate-500 text-sm mt-1">Perturb patient features and observe risk trajectory changes.</p>
        </div>
        {!patient && (
          <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
            No patient loaded — go to Patient Input first.
          </div>
        )}
      </div>

      <div className="grid grid-cols-5 gap-6">
        {/* Left: Controls */}
        <div className="col-span-2 space-y-4">
          <div className="card border border-slate-800 p-5">
            <p className="section-title mb-4">Perturbation Controls</p>

            {grouped.map(group => {
              const gs = GROUP_STYLE[group];
              const sliders = SLIDERS.filter(s => s.group === group);
              return (
                <div key={group} className="mb-5 last:mb-0">
                  <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-800">
                    {gs.icon}
                    <span className="text-xs font-semibold text-slate-300">{gs.label}</span>
                  </div>
                  {sliders.map(s => (
                    <div key={s.key} className="mb-3">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">{s.label} {s.unit && <span className="text-slate-600">({s.unit})</span>}</span>
                        <span className="font-mono text-slate-200">{deltas[s.key] > 0 ? '+' : ''}{deltas[s.key].toFixed(s.step < 1 ? 3 : 1)}</span>
                      </div>
                      <input
                        type="range"
                        min={s.min} max={s.max} step={s.step}
                        value={deltas[s.key]}
                        onChange={e => setDelta(s.key, parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                        style={{ accentColor: gs.color }}
                      />
                      <div className="flex justify-between text-[10px] text-slate-700 mt-0.5">
                        <span>{s.min}</span>
                        <span className="text-slate-600 italic">{s.tip}</span>
                        <span>{s.max}</span>
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          {/* Steps & Actions */}
          <div className="card border border-slate-800 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Trajectory Steps</span>
              <select
                value={nSteps}
                onChange={e => setNSteps(parseInt(e.target.value))}
                className="input-field w-20 text-center"
              >
                {[3, 5, 7, 10].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>

            <button onClick={handleSimulate} disabled={loading || !patient} className="btn-primary w-full justify-center py-2.5 disabled:opacity-50">
              {loading ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
              {loading ? 'Simulating…' : 'Simulate Worsening'}
            </button>

            <button onClick={resetDeltas} className="btn-secondary w-full justify-center py-2 text-sm">
              <RotateCcw size={13} /> Reset Deltas
            </button>

            {error && (
              <div className="px-3 py-2 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right: Results */}
        <div className="col-span-3 space-y-4">
          {!simResult ? (
            <div className="card border border-slate-800 h-full flex items-center justify-center min-h-[400px]">
              <div className="text-center">
                <Zap size={40} className="text-slate-700 mx-auto mb-3" />
                <p className="text-slate-500">Adjust the sliders and click Simulate to see results.</p>
              </div>
            </div>
          ) : (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  {
                    label: 'Risk Category',
                    baseline: simResult.baseline.risk_category,
                    simulated: simResult.simulated.risk_category,
                    color: RISK_COLOR[simResult.simulated.risk_category as keyof typeof RISK_COLOR],
                  },
                  {
                    label: 'Progression Score',
                    baseline: (simResult.baseline.progression_score * 100).toFixed(1) + '%',
                    simulated: (simResult.simulated.progression_score * 100).toFixed(1) + '%',
                    color: simResult.delta_progression > 0 ? '#f43f5e' : '#10b981',
                  },
                ].map((item, i) => (
                  <div key={i} className="card border border-slate-800 p-4">
                    <p className="text-xs text-slate-500 mb-2">{item.label}</p>
                    <div className="flex items-end gap-2">
                      <span className="text-slate-500 text-sm line-through">{item.baseline}</span>
                      <span className="text-lg font-bold" style={{ color: item.color }}>{item.simulated}</span>
                    </div>
                    {i === 1 && (
                      <div className="mt-1">
                        <DeltaBadge value={simResult.delta_progression} />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Trend Chart */}
              <div className="card border border-slate-800 p-5">
                <p className="section-title mb-4">Progression Trajectory ({nSteps} steps)</p>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={trendData}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} unit="%" />
                    <Tooltip
                      formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name === 'progression' ? 'Progression' : 'Risk Score']}
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                    <ReferenceLine y={33} stroke="#10b981" strokeDasharray="4 4" label={{ value: 'Low', fill: '#10b981', fontSize: 10 }} />
                    <ReferenceLine y={67} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: 'High', fill: '#f43f5e', fontSize: 10 }} />
                    <Line type="monotone" dataKey="progression" stroke="#f43f5e" strokeWidth={2} dot={{ fill: '#f43f5e', r: 3 }} name="progression" />
                    <Line type="monotone" dataKey="risk" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 3 }} name="risk" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Before vs After Comparison */}
              <div className="card border border-slate-800 p-5">
                <p className="section-title mb-3">Baseline vs Simulated Modality Scores</p>

                <ResponsiveContainer width="100%" height={140}>
                  <BarChart data={compareData} barSize={18}>
                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip formatter={(v: number) => (v * 100).toFixed(1) + '%'} contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} />
                    <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                    <Bar dataKey="baseline" fill="#3b82f6" name="Baseline" opacity={0.6} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="simulated" fill="#f43f5e" name="Simulated" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>

                <div className="mt-3 border-t border-slate-800 pt-3">
                  <CompareRow label="Clinical Score" baseline={simResult.baseline.clinical_score} simulated={simResult.simulated.clinical_score} />
                  <CompareRow label="Electrical Score" baseline={simResult.baseline.electrical_score} simulated={simResult.simulated.electrical_score} />
                  <CompareRow label="Structural Score" baseline={simResult.baseline.structural_score} simulated={simResult.simulated.structural_score} />
                  <CompareRow label="Progression Score" baseline={simResult.baseline.progression_score} simulated={simResult.simulated.progression_score} />
                </div>
              </div>

              {/* Explanation */}
              <div className="card border border-primary-800/30 bg-primary-900/10 p-5">
                <p className="section-title text-primary-500 mb-2">Clinical Interpretation</p>
                <p className="text-slate-300 text-sm leading-relaxed">{simResult.simulated.explanation}</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
