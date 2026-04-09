# src/visualization/dashboard.py
"""
Visualization dashboard for cardiac digital twin.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

class CardiacTwinDashboard:
    """Interactive dashboard for cardiac digital twin visualization."""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_risk_dashboard(self, fused_profile, risk_score, severity_level):
        """Create comprehensive risk dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Multimodal Risk Profile', 'Risk Score Gauge',
                          'Feature Contributions', 'Risk Over Time'),
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Multimodal risk profile (radar chart style)
        modalities = ['Clinical', 'ECG', 'MRI']
        modality_scores = self._get_modality_scores(fused_profile)
        
        fig.add_trace(
            go.Bar(x=modalities, y=modality_scores, 
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                  text=modality_scores, textposition='auto'),
            row=1, col=1
        )
        
        # 2. Risk score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={'text': "Overall Risk Score (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF6B6B"},
                    'steps': [
                        {'range': [0, 33], 'color': "#90EE90"},
                        {'range': [33, 67], 'color': "#FFD700"},
                        {'range': [67, 100], 'color': "#FF6B6B"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score * 100
                    }
                }
            ),
            row=1, col=2
        )
        
        # 3. Feature contributions
        feature_importance = self._get_feature_importance()
        top_features = feature_importance.head(10)
        
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'],
                  orientation='h', marker_color='#45B7D1'),
            row=2, col=1
        )
        
        # 4. Risk over time (simulated)
        time_points = list(range(10))
        risk_trajectory = self._simulate_risk_trajectory(risk_score)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=risk_trajectory,
                      mode='lines+markers', name='Risk Trajectory',
                      line=dict(color='#FF6B6B', width=3),
                      marker=dict(size=8)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Cardiac Digital Twin - Patient Dashboard",
            height=800,
            showlegend=False,
            title_font_size=20
        )
        
        fig.update_xaxes(title_text="Modality", row=1, col=1)
        fig.update_yaxes(title_text="Risk Score", row=1, col=1)
        fig.update_xaxes(title_text="Importance", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=2)
        fig.update_yaxes(title_text="Risk Score", row=2, col=2)
        
        # Save dashboard
        fig.write_html(self.config.VISUALIZATIONS_DIR / 'risk_dashboard.html')
        fig.write_image(self.config.VISUALIZATIONS_DIR / 'risk_dashboard.png')
        
        return fig
    
    def create_progression_visualization(self, progression_df):
        """Create progression visualization."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Risk Progression', 'Severity Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'pie'}]]
        )
        
        # Risk progression
        fig.add_trace(
            go.Scatter(x=progression_df['time_step'], y=progression_df['risk_score'],
                      mode='lines+markers', name='Risk Score',
                      line=dict(color='#FF6B6B', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # Add severity thresholds
        fig.add_hline(y=0.33, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=0.67, line_dash="dash", line_color="red", row=1, col=1)
        
        # Severity distribution over time
        severity_counts = progression_df['severity_level'].value_counts()
        
        fig.add_trace(
            go.Pie(labels=severity_counts.index, values=severity_counts.values,
                  marker_colors=['#90EE90', '#FFD700', '#FF6B6B']),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Disease Progression Analysis",
            height=500,
            showlegend=True
        )
        
        fig.write_html(self.config.VISUALIZATIONS_DIR / 'progression_dashboard.html')
        
        return fig
    
    def create_multimodal_radar(self, fused_profile):
        """Create multimodal radar chart."""
        # Extract features from each modality
        clinical_features = self._extract_clinical_features(fused_profile)
        ecg_features = self._extract_ecg_features(fused_profile)
        mri_features = self._extract_mri_features(fused_profile)
        
        # Normalize features
        all_features = {**clinical_features, **ecg_features, **mri_features}
        categories = list(all_features.keys())
        values = list(all_features.values())
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Patient Profile',
            line_color='#FF6B6B'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Multimodal Cardiac Profile"
        )
        
        fig.write_html(self.config.VISUALIZATIONS_DIR / 'multimodal_radar.html')
        
        return fig
    
    def create_comparison_plot(self, comparison_df):
        """Create model comparison plot."""
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Model Performance Comparison',)
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig.add_trace(
                go.Bar(name=metric, x=comparison_df['Modality'], y=comparison_df[metric],
                      text=comparison_df[metric].round(3), textposition='auto'),
                row=1, col=1
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            yaxis_title="Score",
            height=600
        )
        
        fig.write_html(self.config.VISUALIZATIONS_DIR / 'model_comparison.html')
        
        return fig
    
    def create_feature_importance_plot(self, importance_df, top_n=20):
        """Create feature importance visualization."""
        top_features = importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            )
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        fig.write_html(self.config.VISUALIZATIONS_DIR / 'feature_importance.html')
        
        return fig
    
    def _get_modality_scores(self, fused_profile):
        """Extract risk scores per modality."""
        clinical_score = 0
        ecg_score = 0
        mri_score = 0
        
        for key, value in fused_profile.items():
            if 'clinical' in key or key in ['age', 'sex', 'cp', 'trestbps', 'chol']:
                clinical_score += value if isinstance(value, (int, float)) else 0
            elif 'ecg' in key or key in ['heart_rate', 'rr_mean', 'abnormality_score']:
                ecg_score += value if isinstance(value, (int, float)) else 0
            elif 'mri' in key or key in ['lv_area', 'ejection_fraction', 'dysfunction_score']:
                mri_score += value if isinstance(value, (int, float)) else 0
        
        # Normalize scores
        total = clinical_score + ecg_score + mri_score
        if total > 0:
            clinical_score = clinical_score / total
            ecg_score = ecg_score / total
            mri_score = mri_score / total
        
        return [clinical_score, ecg_score, mri_score]
    
    def _get_feature_importance(self):
        """Load feature importance from saved file."""
        try:
            importance_df = pd.read_csv(self.config.RESULTS_DIR / 'feature_importance.csv')
            return importance_df
        except FileNotFoundError:
            # Return dummy data if file not found
            return pd.DataFrame({
                'feature': ['feature1', 'feature2', 'feature3'],
                'importance': [0.3, 0.2, 0.1]
            })
    
    def _simulate_risk_trajectory(self, initial_risk, n_steps=10):
        """Simulate risk trajectory over time."""
        # Simple exponential growth model
        trajectory = []
        for t in range(n_steps):
            risk = min(1.0, initial_risk * np.exp(0.1 * t))
            trajectory.append(risk)
        return trajectory
    
    def _extract_clinical_features(self, profile):
        """Extract clinical features from profile."""
        clinical = {}
        for key, value in profile.items():
            if 'clinical' in key or key in ['age', 'sex', 'cp', 'trestbps', 'chol']:
                if isinstance(value, (int, float)):
                    clinical[key] = value
        return self._normalize_features(clinical)
    
    def _extract_ecg_features(self, profile):
        """Extract ECG features from profile."""
        ecg = {}
        for key, value in profile.items():
            if 'ecg' in key or key in ['heart_rate', 'rr_mean', 'abnormality_score']:
                if isinstance(value, (int, float)):
                    ecg[key] = value
        return self._normalize_features(ecg)
    
    def _extract_mri_features(self, profile):
        """Extract MRI features from profile."""
        mri = {}
        for key, value in profile.items():
            if 'mri' in key or key in ['lv_area', 'ejection_fraction', 'dysfunction_score']:
                if isinstance(value, (int, float)):
                    mri[key] = value
        return self._normalize_features(mri)
    
    def _normalize_features(self, features):
        """Normalize features to 0-1 range."""
        if not features:
            return features
        
        values = list(features.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val > min_val:
            for key in features:
                features[key] = (features[key] - min_val) / (max_val - min_val)
        else:
            for key in features:
                features[key] = 0.5
        
        return features