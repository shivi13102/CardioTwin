// src/App.tsx — Main application with sidebar navigation

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Heart, LayoutDashboard, User, Activity, Zap, ChevronRight,
  Stethoscope, BrainCircuit, Database
} from 'lucide-react';
import type { Page, PatientInput, PredictResult, SimulateResult } from './types';
import Dashboard from './pages/Dashboard';
import PatientInputPage from './pages/PatientInput';
import ResultsPage from './pages/Results';
import SimulationPage from './pages/Simulation';
import DataInfoPage from './pages/DataInfo';

interface NavItem { id: Page; label: string; icon: React.ReactNode; }

const NAV_ITEMS: NavItem[] = [
  { id: 'dashboard',  label: 'Dashboard',     icon: <LayoutDashboard size={18} /> },
  { id: 'datainfo',   label: 'Data Info',     icon: <Database size={18} /> },
  { id: 'patient',    label: 'Patient Input',  icon: <User size={18} /> },
  { id: 'results',    label: 'Twin Analysis',  icon: <BrainCircuit size={18} /> },
  { id: 'simulation', label: 'Simulation',     icon: <Zap size={18} /> },
];

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [patient, setPatient] = useState<PatientInput | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [simResult, setSimResult] = useState<SimulateResult | null>(null);

  const navigateTo = (p: Page) => setPage(p);

  const handlePatientLoaded = (p: PatientInput) => {
    setPatient(p);
    setPage('patient');
  };

  const handleResultReady = (p: PatientInput, r: PredictResult) => {
    setPatient(p);
    setResult(r);
    setPage('results');
  };

  const handleSimDone = (s: SimulateResult) => {
    setSimResult(s);
    setPage('simulation');
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      {/* ── Sidebar ── */}
      <aside className="w-64 flex-shrink-0 bg-slate-900/80 border-r border-slate-800 flex flex-col backdrop-blur-sm">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary-600 flex items-center justify-center shadow-lg shadow-primary-600/30">
              <Heart size={18} className="text-white" fill="currentColor" />
            </div>
            <div>
              <p className="font-bold text-base text-white tracking-tight">CardioTwin</p>
              <p className="text-[10px] text-slate-500 font-medium tracking-wide">CARDIAC DIGITAL TWIN</p>
            </div>
          </div>
        </div>

        {/* Status badge */}
        <div className="px-4 py-3 border-b border-slate-800">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-400">Cardiac-FM Active</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <p className="section-title px-2 mb-4">Navigation</p>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => navigateTo(item.id)}
              className={`nav-link w-full text-left${page === item.id ? ' nav-link-active' : ''}`}
            >
              {item.icon}
              <span>{item.label}</span>
              {page === item.id && (
                <ChevronRight size={14} className="ml-auto text-primary-400" />
              )}
            </button>
          ))}
        </nav>

        {/* Footer info */}
        <div className="px-4 py-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Stethoscope size={13} />
            <span>Research Prototype · v1.0</span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1 leading-tight">
            Not for clinical use. For demo only.
          </p>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="h-full"
          >
            {page === 'dashboard' && (
              <Dashboard
                onLoadPatient={handlePatientLoaded}
                onNavigate={navigateTo}
              />
            )}
            {page === 'patient' && (
              <PatientInputPage
                initialPatient={patient}
                onResultReady={handleResultReady}
              />
            )}
            {page === 'results' && (
              <ResultsPage
                result={result}
                patient={patient}
                onSimulate={() => navigateTo('simulation')}
                onBack={() => navigateTo('patient')}
              />
            )}
            {page === 'simulation' && (
              <SimulationPage
                patient={patient}
                baselineResult={result}
                simResult={simResult}
                onSimDone={handleSimDone}
                onBack={() => navigateTo('results')}
              />
            )}
            {page === 'datainfo' && (
              <DataInfoPage />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
