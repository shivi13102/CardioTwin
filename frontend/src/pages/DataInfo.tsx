import { motion } from 'framer-motion';
import { Database, Heart, Activity, Scan, Network } from 'lucide-react';

interface DatasetMeta {
  title: string;
  icon: React.ReactNode;
  description: string;
  samples: string;
  features: string;
  label: string;
  color: string;
  plots: Array<{ path: string; title: string; desc: string }>;
}

const DATASETS: DatasetMeta[] = [
  {
    title: 'Clinical (EHR) Cohort',
    icon: <Heart size={20} className="text-blue-400" />,
    description: 'Routine clinical parameters, demographics, and blood laboratory results such as cholesterol and fasting blood sugar.',
    samples: '~303',
    features: '14 original (Age, Trestbps, Chol, etc.)',
    label: 'num',
    color: 'border-blue-500/30',
    plots: [
      { path: '/visualization/ehr/ehr_class_distribution.png', title: 'Disease Class Distribution', desc: 'Shows how samples are distributed across severity categories.' },
      { path: '/visualization/ehr/ehr_violin_plot.png', title: 'Feature Variation Across Classes', desc: 'Illustrates how important biomarkers vary across disease burden.' },
      { path: '/visualization/ehr/ehr_projection.png', title: 'Feature Space Projection', desc: 'Visualizes whether samples separate in lower-dimensional feature space.' }
    ]
  },
  {
    title: 'Electrophysiological (ECG) Cohort',
    icon: <Activity size={20} className="text-amber-400" />,
    description: 'High-frequency continuous waveform metrics including heart rate variability, spectral power, and morphological features.',
    samples: '~10,000+',
    features: '42 extracted wave features',
    label: 'severity_group',
    color: 'border-amber-500/30',
    plots: [
      { path: '/visualization/ecg/ecg_class_distribution.png', title: 'Severity Group Distribution', desc: 'Shows how samples are distributed across severity categories.' },
      { path: '/visualization/ecg/ecg_violin_plot.png', title: 'Biomarker Variation', desc: 'Illustrates how important biomarkers vary across disease burden.' },
      { path: '/visualization/ecg/ecg_projection.png', title: 'Feature Space Projection', desc: 'Visualizes whether samples separate in lower-dimensional feature space.' }
    ]
  },
  {
    title: 'Structural (MRI) Cohort',
    icon: <Scan size={20} className="text-purple-400" />,
    description: 'Biventricular structural quantifications including mass, volumes, eccentricity, and global intensity entropy.',
    samples: '~5,000+',
    features: '26 structural markers',
    label: 'severity_group / severity_score',
    color: 'border-purple-500/30',
    plots: [
      { path: '/visualization/mri/mri_class_distribution.png', title: 'Severity Group Distribution', desc: 'Shows how samples are distributed across severity categories.' },
      { path: '/visualization/mri/mri_violin_plot.png', title: 'Structural Variation', desc: 'Illustrates how important biomarkers vary across disease burden.' },
      { path: '/visualization/mri/mri_projection.png', title: 'Feature Space Projection', desc: 'Visualizes whether samples separate in lower-dimensional feature space.' }
    ]
  },
  {
    title: 'Prototype Multimodal Fusion',
    icon: <Network size={20} className="text-emerald-400" />,
    description: 'A synthetic prototype combining representative features from EHR, ECG, and MRI into a single unified space for methodological exploration. Not representative of subject-matched cross-clinical alignment.',
    samples: 'Truncated fusion',
    features: '14 fused proxies',
    label: 'Prototype Extrapolated Label',
    color: 'border-emerald-500/30',
    plots: [
      { path: '/visualization/multimodal/multimodal_class_comparison.png', title: 'Severity Distribution Comparison', desc: 'Shows how samples are distributed across severity categories.' },
      { path: '/visualization/multimodal/multimodal_scatter.png', title: 'ECG Instability vs MRI Structural Burden', desc: 'Highlights cross-feature relationships associated with structural or electrophysiological burden.' },
      { path: '/visualization/multimodal/multimodal_projection.png', title: 'Fused Projection Space', desc: 'Visualizes whether samples separate in lower-dimensional feature space.' }
    ]
  }
];

export default function DataInfoPage() {
  return (
    <div className="min-h-full p-8 max-w-6xl mx-auto space-y-10 pb-20">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
          <Database className="text-primary-500" />
          Datasets & Exploratory Analysis
        </h1>
        <p className="text-slate-400 max-w-3xl">
          Visualizations generated directly from the underlying training data providing insight into individual dataset characteristics and prototype fusion validity.
        </p>
        <div className="mt-4 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl max-w-3xl">
          <p className="text-amber-400 text-sm font-medium">Research Disclaimer</p>
          <p className="text-amber-400/80 text-xs mt-1 leading-relaxed">
            The EHR, ECG, and MRI datasets utilized in this framework originated from separate unsynchronized clinical sources. Thus, the multimodal visualizations herein represent a prototype-level fused representation intended only for generalized methodological exploration. They do not indicate subject-level matched clinical correlation.
          </p>
        </div>
      </div>

      <div className="space-y-12">
        {DATASETS.map((dataset, idx) => (
          <motion.div 
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className={`card border ${dataset.color} p-0 overflow-hidden`}
          >
            {/* Dataset Header Card */}
            <div className="p-6 bg-slate-900/50 border-b border-slate-800">
              <div className="flex items-center gap-3 mb-3">
                {dataset.icon}
                <h2 className="text-xl font-bold text-white">{dataset.title}</h2>
              </div>
              <p className="text-slate-400 text-sm mb-4 max-w-3xl">{dataset.description}</p>
              
              <div className="flex flex-wrap gap-4 text-xs">
                <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 flex items-center gap-2">
                  <span className="text-slate-500">Samples:</span>
                  <span className="text-slate-300 font-mono">{dataset.samples}</span>
                </div>
                <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 flex items-center gap-2">
                  <span className="text-slate-500">Features:</span>
                  <span className="text-slate-300 font-mono">{dataset.features}</span>
                </div>
                <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 flex items-center gap-2">
                  <span className="text-slate-500">Target Label:</span>
                  <span className="text-primary-400 font-mono">{dataset.label}</span>
                </div>
              </div>
            </div>

            {/* Generated Plots Display */}
            <div className="p-6">
              <h3 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">Key Visualizations</h3>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {dataset.plots.map((plot, pIdx) => (
                  <div key={pIdx} className="flex flex-col h-full bg-slate-950 rounded-xl border border-slate-800/80 p-3">
                    <div className="w-full aspect-[4/3] relative flex items-center justify-center bg-slate-900 rounded-lg overflow-hidden mb-3 group">
                      <img 
                        src={plot.path} 
                        alt={plot.title}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          (e.currentTarget.parentElement as HTMLElement).innerHTML = `<div class="text-center p-4"><div class="text-slate-600 text-[10px] mb-2 font-mono break-all">${plot.path}</div><span class="text-slate-500 text-xs">Image not generated yet. Please run:<br/><code class="text-primary-400/70 mt-1 block">python src/visualization/generate_visualizations.py</code></span></div>`;
                        }}
                      />
                    </div>
                    <div className="mt-auto">
                      <h4 className="text-slate-200 text-sm font-medium mb-1">{plot.title}</h4>
                      <p className="text-slate-500 text-xs leading-relaxed">{plot.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
