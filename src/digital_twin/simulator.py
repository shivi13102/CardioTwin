# src/digital_twin/simulator.py
"""
Digital twin simulation layer for disease progression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class DigitalTwinSimulator:
    """Simulate disease progression and intervention effects."""
    
    def __init__(self, config, multimodal_model):
        self.config = config
        self.model = multimodal_model
        self.progression_models = {}
        
    def simulate_progression(self, fused_profile, time_steps=10):
        """Simulate disease progression over time."""
        print("\n=== Simulating Disease Progression ===")
        
        # Get baseline features
        features = self._extract_features(fused_profile)
        baseline_risk = self._predict_risk(features)
        
        # Simulate progression
        progression_trajectory = []
        
        for t in range(time_steps):
            # Apply progression factors
            progressed_features = self._apply_progression(features, t)
            
            # Predict risk at this time point
            risk = self._predict_risk(progressed_features)
            
            progression_trajectory.append({
                'time_step': t,
                'risk_score': risk,
                'severity_level': self._risk_to_severity(risk)
            })
        
        progression_df = pd.DataFrame(progression_trajectory)
        
        # Plot progression
        self._plot_progression(progression_df)
        
        return progression_df
    
    def simulate_intervention(self, fused_profile, intervention_type='medication', 
                            intensity=0.3, time_steps=10):
        """Simulate effect of intervention on disease progression."""
        print(f"\n=== Simulating {intervention_type} Intervention ===")
        
        # Get baseline features
        features = self._extract_features(fused_profile)
        baseline_risk = self._predict_risk(features)
        
        # Simulate progression with and without intervention
        progression_no_intervention = []
        progression_with_intervention = []
        
        for t in range(time_steps):
            # Without intervention
            features_no_intervention = self._apply_progression(features, t)
            risk_no_intervention = self._predict_risk(features_no_intervention)
            
            # With intervention
            features_with_intervention = self._apply_intervention(
                self._apply_progression(features, t), 
                intervention_type, intensity
            )
            risk_with_intervention = self._predict_risk(features_with_intervention)
            
            progression_no_intervention.append({
                'time_step': t,
                'risk_score': risk_no_intervention
            })
            
            progression_with_intervention.append({
                'time_step': t,
                'risk_score': risk_with_intervention
            })
        
        # Create DataFrames
        df_no_intervention = pd.DataFrame(progression_no_intervention)
        df_with_intervention = pd.DataFrame(progression_with_intervention)
        
        # Plot comparison
        self._plot_intervention_comparison(df_no_intervention, df_with_intervention, 
                                          intervention_type)
        
        # Calculate intervention benefit
        final_risk_no_intervention = df_no_intervention.iloc[-1]['risk_score']
        final_risk_with_intervention = df_with_intervention.iloc[-1]['risk_score']
        risk_reduction = final_risk_no_intervention - final_risk_with_intervention
        
        print(f"\nIntervention Benefit:")
        print(f"Final risk without intervention: {final_risk_no_intervention:.3f}")
        print(f"Final risk with intervention: {final_risk_with_intervention:.3f}")
        print(f"Risk reduction: {risk_reduction:.3f}")
        print(f"Relative risk reduction: {(risk_reduction / final_risk_no_intervention * 100):.1f}%")
        
        return df_no_intervention, df_with_intervention
    
    def simulate_scenario(self, fused_profile, scenario_changes, time_steps=10):
        """Simulate specific scenario with custom changes."""
        print("\n=== Simulating Custom Scenario ===")
        
        print(f"Scenario changes: {scenario_changes}")
        
        # Get baseline features
        features = self._extract_features(fused_profile)
        
        # Simulate progression
        progression = []
        
        for t in range(time_steps):
            # Apply progression
            progressed_features = self._apply_progression(features, t)
            
            # Apply scenario-specific changes
            for feature, change in scenario_changes.items():
                if feature in progressed_features:
                    progressed_features[feature] += change
            
            # Predict risk
            risk = self._predict_risk(progressed_features)
            
            progression.append({
                'time_step': t,
                'risk_score': risk,
                'severity_level': self._risk_to_severity(risk)
            })
        
        progression_df = pd.DataFrame(progression)
        
        # Plot scenario progression
        self._plot_scenario_progression(progression_df, scenario_changes)
        
        return progression_df
    
    def _extract_features(self, fused_profile):
        """Extract features from fused profile."""
        # Convert profile to DataFrame
        if isinstance(fused_profile, dict):
            profile_df = pd.DataFrame([fused_profile])
        elif isinstance(fused_profile, pd.Series):
            profile_df = pd.DataFrame([fused_profile.to_dict()])
        else:
            profile_df = fused_profile
        
        # Select only features used by model
        feature_cols = [col for col in self.model.feature_names if col in profile_df.columns]
        
        if not feature_cols:
            raise ValueError("No matching features found in profile")
        
        features = profile_df[feature_cols]
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Normalize if scaler available
        if hasattr(self.model, 'scaler'):
            features_scaled = self.model.scaler.transform(features)
            features = pd.DataFrame(features_scaled, columns=feature_cols)
        
        return features
    
    def _predict_risk(self, features):
        """Predict risk score for features."""
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.values.reshape(1, -1)
        
        # Predict
        if hasattr(self.model.model, 'predict_proba'):
            # Get probability of high risk class
            proba = self.model.model.predict_proba(features)
            # For multi-class, use probability of highest risk class
            risk = proba[0][-1]  # Last class is highest risk
        else:
            # Use decision function or predict
            risk = self.model.model.predict(features)[0]
        
        return risk
    
    def _apply_progression(self, features, time_step):
        """Apply natural disease progression over time."""
        # Create a copy
        progressed = features.copy()
        
        # Apply progression to relevant features
        for col in progressed.columns:
            if 'clinical' in col or 'ecg' in col or 'mri' in col:
                # Different progression rates for different features
                if 'oldpeak' in col or 'abnormality' in col or 'dysfunction' in col:
                    # Fast progression for pathological features
                    progression_rate = 0.05
                elif 'age' in col:
                    progression_rate = 0.01
                else:
                    progression_rate = 0.02
                
                # Apply progression (increase risk factors)
                if progressed[col].dtype in [np.float64, np.int64]:
                    progressed[col] = progressed[col] * (1 + progression_rate * time_step)
        
        return progressed
    
    def _apply_intervention(self, features, intervention_type, intensity):
        """Apply intervention effects to features."""
        # Create a copy
        intervened = features.copy()
        
        if intervention_type == 'medication':
            # Medication reduces risk factors
            for col in intervened.columns:
                if 'bp' in col or 'chol' in col:
                    intervened[col] = intervened[col] * (1 - intensity * 0.5)
                elif 'oldpeak' in col:
                    intervened[col] = intervened[col] * (1 - intensity * 0.7)
        
        elif intervention_type == 'lifestyle':
            # Lifestyle changes improve multiple factors
            for col in intervened.columns:
                if 'bp' in col or 'chol' in col or 'oldpeak' in col:
                    intervened[col] = intervened[col] * (1 - intensity * 0.3)
        
        elif intervention_type == 'surgical':
            # Surgical intervention for structural issues
            for col in intervened.columns:
                if 'mri' in col and ('ejection_fraction' in col or 'dysfunction' in col):
                    intervened[col] = intervened[col] * (1 - intensity * 0.8)
        
        elif intervention_type == 'device':
            # Device therapy for electrical issues
            for col in intervened.columns:
                if 'ecg' in col and ('abnormality' in col or 'rr' in col):
                    intervened[col] = intervened[col] * (1 - intensity * 0.6)
        
        return intervened
    
    def _risk_to_severity(self, risk_score):
        """Convert risk score to severity level."""
        if risk_score < 0.33:
            return "Low Risk"
        elif risk_score < 0.67:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def _plot_progression(self, progression_df):
        """Plot disease progression over time."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(progression_df['time_step'], progression_df['risk_score'], 
                'b-o', linewidth=2, markersize=8)
        
        # Add severity thresholds
        plt.axhline(y=0.33, color='g', linestyle='--', label='Low/Moderate Threshold')
        plt.axhline(y=0.67, color='r', linestyle='--', label='Moderate/High Threshold')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Risk Score')
        plt.title('Disease Progression Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add severity zones
        plt.fill_between([0, max(progression_df['time_step'])], 0, 0.33, 
                        alpha=0.2, color='green', label='Low Risk Zone')
        plt.fill_between([0, max(progression_df['time_step'])], 0.33, 0.67, 
                        alpha=0.2, color='yellow', label='Moderate Risk Zone')
        plt.fill_between([0, max(progression_df['time_step'])], 0.67, 1, 
                        alpha=0.2, color='red', label='High Risk Zone')
        
        plt.tight_layout()
        plt.savefig(self.config.VISUALIZATIONS_DIR / 'disease_progression.png')
        plt.close()
    
    def _plot_intervention_comparison(self, df_no_intervention, df_with_intervention, 
                                      intervention_type):
        """Plot comparison with and without intervention."""
        plt.figure(figsize=(12, 6))
        
        plt.plot(df_no_intervention['time_step'], df_no_intervention['risk_score'], 
                'r-o', linewidth=2, markersize=8, label='No Intervention')
        plt.plot(df_with_intervention['time_step'], df_with_intervention['risk_score'], 
                'g-s', linewidth=2, markersize=8, label=f'With {intervention_type.capitalize()}')
        
        # Add severity thresholds
        plt.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.67, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Risk Score')
        plt.title(f'Effect of {intervention_type.capitalize()} Intervention on Disease Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display benefit
        final_risk_no = df_no_intervention.iloc[-1]['risk_score']
        final_risk_with = df_with_intervention.iloc[-1]['risk_score']
        benefit = final_risk_no - final_risk_with
        plt.text(0.05, 0.95, f'Risk Reduction: {benefit:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.config.VISUALIZATIONS_DIR / f'intervention_{intervention_type}.png')
        plt.close()
    
    def _plot_scenario_progression(self, progression_df, scenario_changes):
        """Plot progression under custom scenario."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(progression_df['time_step'], progression_df['risk_score'], 
                'b-o', linewidth=2, markersize=8)
        
        # Add severity thresholds
        plt.axhline(y=0.33, color='g', linestyle='--', label='Low/Moderate Threshold')
        plt.axhline(y=0.67, color='r', linestyle='--', label='Moderate/High Threshold')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Risk Score')
        
        # Create scenario description
        scenario_text = "\n".join([f"{k}: {v}" for k, v in scenario_changes.items()])
        plt.title(f'Scenario Simulation\nChanges: {scenario_text[:50]}...')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add severity zones
        plt.fill_between([0, max(progression_df['time_step'])], 0, 0.33, 
                        alpha=0.2, color='green')
        plt.fill_between([0, max(progression_df['time_step'])], 0.33, 0.67, 
                        alpha=0.2, color='yellow')
        plt.fill_between([0, max(progression_df['time_step'])], 0.67, 1, 
                        alpha=0.2, color='red')
        
        plt.tight_layout()
        plt.savefig(self.config.VISUALIZATIONS_DIR / 'scenario_simulation.png')
        plt.close()