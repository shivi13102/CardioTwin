# src/preprocessing/ecg_preprocessor.py
"""
ECG data preprocessing module - OPTIMIZED VERSION
"""

import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from scipy.signal import find_peaks, welch
import pywt
import warnings
from functools import lru_cache
warnings.filterwarnings('ignore')

class ECGPreprocessor:
    """Preprocessor for ECG (MIT-BIH) dataset - Optimized for speed."""
    
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config.ECG_SAMPLING_RATE
        self.segment_length = config.ECG_SEGMENT_LENGTH
        
        # Pre-compute filter coefficients (reuse across records)
        self._init_filters()
        
    def _init_filters(self):
        """Pre-compute filter coefficients for reuse."""
        nyquist = self.sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 45 / nyquist
        
        # Bandpass filter
        self.b, self.a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        
        # Median filter window size
        self.median_window = self._ensure_odd(int(0.2 * self.sampling_rate))
        
    def _ensure_odd(self, x):
        """Ensure a number is odd."""
        return x if x % 2 == 1 else x + 1
    
    def load_signal(self, filepath, record_name):
        """Load ECG signal from MIT-BIH database."""
        from pathlib import Path
        record_path = str(Path(filepath) / record_name)
        record = wfdb.rdrecord(record_path)
        signal_data = record.p_signal
        
        # Use first lead if multiple leads available
        if signal_data.shape[1] > 1:
            signal_data = signal_data[:, 0]
        else:
            signal_data = signal_data[:, 0]
        
        # Truncate to first 30 seconds to avoid slow processing of 30-min records
        max_samples = int(30 * record.fs)  # 30 seconds
        if len(signal_data) > max_samples:
            signal_data = signal_data[:max_samples]
        
        return signal_data, record.fs
    
    def remove_noise(self, signal_data):
        """Remove noise from ECG signal - optimized with pre-computed filters."""
        # Apply bandpass filter
        filtered_signal = signal.filtfilt(self.b, self.a, signal_data)
        
        # Remove baseline wander using median filter
        baseline = signal.medfilt(filtered_signal, kernel_size=self.median_window)
        cleaned_signal = filtered_signal - baseline
        
        return cleaned_signal
    
    def detect_r_peaks(self, signal_data):
        """Detect R peaks in ECG signal - optimized."""
        # Derivative
        diff_signal = np.diff(signal_data)
        
        # Square the signal
        squared_signal = diff_signal ** 2
        
        # Moving average window
        window_size = int(0.15 * self.sampling_rate)
        window = np.ones(window_size) / window_size
        ma_signal = np.convolve(squared_signal, window, mode='same')
        
        # Find peaks
        threshold = 0.5 * np.max(ma_signal)
        peaks, _ = find_peaks(ma_signal, 
                              height=threshold,
                              distance=int(0.3 * self.sampling_rate))
        
        # Vectorized peak refinement
        if len(peaks) > 0:
            refined_peaks = []
            for peak in peaks:
                window_start = max(0, peak - 10)
                window_end = min(len(signal_data), peak + 10)
                refined_peaks.append(window_start + np.argmax(signal_data[window_start:window_end]))
            return np.array(refined_peaks)
        
        return np.array([])
    
    def calculate_rr_intervals(self, r_peaks):
        """Calculate RR intervals from R peaks."""
        if len(r_peaks) < 2:
            return np.array([])
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # in ms
        return rr_intervals
    
    def extract_time_domain_features(self, signal_data, r_peaks, rr_intervals):
        """Extract time-domain ECG features - optimized."""
        features = {}
        
        # Heart rate metrics
        if len(rr_intervals) > 0:
            mean_rr = np.mean(rr_intervals)
            features['heart_rate'] = 60000 / mean_rr
            features['rr_mean'] = mean_rr
            features['rr_std'] = np.std(rr_intervals)
            features['rr_min'] = np.min(rr_intervals)
            features['rr_max'] = np.max(rr_intervals)
            
            # HRV metrics
            diff_rr = np.diff(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(diff_rr ** 2))
            features['sdnn'] = features['rr_std']
            
            # pNN50
            nn50 = np.sum(np.abs(diff_rr) > 50)
            features['pnn50'] = nn50 / len(rr_intervals) * 100
            
            # HRV index
            rr_hist, _ = np.histogram(rr_intervals, bins=20)
            features['hrv_index'] = np.sum(rr_hist > 0)
        else:
            # Default values
            features.update({
                'heart_rate': 0, 'rr_mean': 0, 'rr_std': 0, 'rr_min': 0,
                'rr_max': 0, 'rmssd': 0, 'sdnn': 0, 'pnn50': 0, 'hrv_index': 0
            })
            mean_rr = 0
        
        # Morphological features
        if len(r_peaks) > 2:
            peak_indices = r_peaks[:5]
            features['r_amplitude'] = np.mean(signal_data[peak_indices])
            
            # T wave amplitude (vectorized)
            t_windows = peak_indices + int(0.2 * self.sampling_rate)
            t_windows = t_windows[t_windows < len(signal_data)]
            if len(t_windows) > 0:
                features['t_amplitude'] = np.mean(signal_data[t_windows])
            else:
                features['t_amplitude'] = 0
            
            features['qrs_duration'] = 80  # Placeholder
        else:
            features.update({'qrs_duration': 80, 'r_amplitude': 0, 't_amplitude': 0})
        
        # Statistical features (fast)
        features['signal_mean'] = np.mean(signal_data)
        features['signal_std'] = np.std(signal_data)
        features['signal_skew'] = self._calculate_skewness(signal_data)
        features['signal_kurtosis'] = self._calculate_kurtosis(signal_data)
        
        # Sample entropy - can be skipped for speed
        if len(signal_data) > 1000:
            features['sample_entropy'] = self._calculate_entropy_fast(signal_data)
        else:
            features['sample_entropy'] = 0
        
        return features
    
    def extract_frequency_domain_features(self, signal_data):
        """Extract frequency-domain ECG features - optimized."""
        features = {}
        
        # Use fewer points for faster FFT
        nperseg = min(256, len(signal_data) // 4)
        if nperseg < 8:
            nperseg = len(signal_data)
        
        # Compute power spectral density
        frequencies, psd = welch(signal_data, self.sampling_rate, 
                                 nperseg=nperseg, noverlap=nperseg//2)
        
        # Define frequency bands
        vlf_band = (0.0033, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate power in each band (vectorized)
        vlf_mask = (frequencies >= vlf_band[0]) & (frequencies < vlf_band[1])
        lf_mask = (frequencies >= lf_band[0]) & (frequencies < lf_band[1])
        hf_mask = (frequencies >= hf_band[0]) & (frequencies < hf_band[1])
        
        features['vlf_power'] = np.trapz(psd[vlf_mask], frequencies[vlf_mask]) if np.any(vlf_mask) else 0
        features['lf_power'] = np.trapz(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
        features['hf_power'] = np.trapz(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
        
        # LF/HF ratio
        features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-10)
        
        # Total power
        features['total_power'] = np.trapz(psd, frequencies)
        
        # Normalized LF and HF
        total = features['vlf_power'] + features['lf_power'] + features['hf_power']
        features['lf_norm'] = features['lf_power'] / (total + 1e-10) * 100
        features['hf_norm'] = features['hf_power'] / (total + 1e-10) * 100
        
        return features
    
    def extract_wavelet_features(self, signal_data):
        """Extract wavelet-based features - optimized."""
        features = {}
        
        # Reduce signal length if too long
        if len(signal_data) > 5000:
            signal_data = signal_data[::2]  # Downsample by 2 for wavelet analysis
        
        # Perform wavelet decomposition
        try:
            coeffs = pywt.wavedec(signal_data, 'db4', level=4)
        except:
            # Fallback if decomposition fails
            coeffs = [signal_data]
        
        # Extract energy and entropy
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff ** 2)
            features[f'wavelet_energy_level_{i}'] = energy
            
            if len(coeff) > 0 and energy > 0:
                prob = (coeff ** 2) / (energy + 1e-10)
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                features[f'wavelet_entropy_level_{i}'] = entropy
            else:
                features[f'wavelet_entropy_level_{i}'] = 0
        
        # Total wavelet energy
        features['total_wavelet_energy'] = sum(np.sum(c ** 2) for c in coeffs)
        
        return features
    
    def extract_rhythm_features(self, signal_data, r_peaks, rr_intervals):
        """Extract rhythm-related features - optimized."""
        features = {}
        
        # Calculate arrhythmia indicators
        if len(rr_intervals) > 0:
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            features['rr_cv'] = std_rr / (mean_rr + 1e-10)
            
            if len(rr_intervals) > 1:
                features['irregularity_score'] = np.std(np.diff(rr_intervals)) / (mean_rr + 1e-10)
            else:
                features['irregularity_score'] = 0
            
            # Ectopic beats
            rr_diff = np.diff(rr_intervals)
            ectopic_threshold = 0.2 * mean_rr
            ectopic_beats = np.sum(np.abs(rr_diff) > ectopic_threshold)
            features['ectopic_beat_count'] = ectopic_beats
            features['ectopic_percentage'] = ectopic_beats / len(rr_intervals) * 100
        else:
            features.update({
                'rr_cv': 0, 'irregularity_score': 0,
                'ectopic_beat_count': 0, 'ectopic_percentage': 0
            })
        
        # Rhythm stability
        features['rhythm_stability'] = 1 / (features.get('rr_cv', 1) + 0.1)
        
        # Abnormality severity
        abnormality_score = 0
        if features.get('rr_cv', 0) > 0.1:
            abnormality_score += 1
        if features.get('ectopic_percentage', 0) > 5:
            abnormality_score += 1
        if features.get('lf_hf_ratio', 1) > 2 or features.get('lf_hf_ratio', 1) < 0.5:
            abnormality_score += 1
        
        features['abnormality_score'] = abnormality_score
        
        # Severity group
        if abnormality_score == 0:
            features['severity_group'] = 'normal'
        elif abnormality_score <= 2:
            features['severity_group'] = 'mild'
        else:
            features['severity_group'] = 'severe'
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness - optimized."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum((data - mean) ** 3) / (n * std ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis - optimized."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum((data - mean) ** 4) / (n * std ** 4) - 3
    
    def _calculate_entropy_fast(self, data, m=2, r=0.2):
        """Fast sample entropy calculation using vectorization."""
        if len(data) < m + 1:
            return 0
        
        # Scale r to standard deviation
        r = r * np.std(data)
        
        def _phi(m):
            N = len(data) - m + 1
            # Create embedding matrix
            vectors = np.array([data[i:i+m] for i in range(N)])
            
            # Calculate distances efficiently using broadcasting
            # This is still O(N^2) but vectorized
            distances = np.zeros((N, N))
            for i in range(m):
                distances += (vectors[:, i:i+1] - vectors[:, i]) ** 2
            distances = np.sqrt(distances)
            
            # Count matches (excluding self)
            matches = np.sum(distances <= r, axis=1) - 1
            return np.sum(matches)
        
        phi_m = _phi(m)
        if phi_m == 0:
            return 0
        
        phi_m1 = _phi(m + 1)
        if phi_m1 == 0:
            return 0
        
        return -np.log(phi_m1 / phi_m)
    
    def process_record(self, filepath, record_name):
        """Process a single ECG record - optimized."""
        # Load signal
        signal_data, fs = self.load_signal(filepath, record_name)
        
        # Check sampling rate
        if fs != self.sampling_rate:
            resampled_length = int(len(signal_data) * self.sampling_rate / fs)
            signal_data = signal.resample(signal_data, resampled_length)
        
        # Remove noise
        cleaned_signal = self.remove_noise(signal_data)
        
        # Detect R peaks
        r_peaks = self.detect_r_peaks(cleaned_signal)
        
        # Calculate RR intervals
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        
        # Extract features
        time_features = self.extract_time_domain_features(cleaned_signal, r_peaks, rr_intervals)
        freq_features = self.extract_frequency_domain_features(cleaned_signal)
        wavelet_features = self.extract_wavelet_features(cleaned_signal)
        rhythm_features = self.extract_rhythm_features(cleaned_signal, r_peaks, rr_intervals)
        
        # Combine all features
        all_features = {**time_features, **freq_features, **wavelet_features, **rhythm_features}
        
        return all_features, cleaned_signal, r_peaks
    
    def process_all_records(self, filepath, record_list):
        """Process all ECG records in list - optimized with progress bar."""
        all_features = []
        failed_records = []
        
        from pathlib import Path
        from tqdm import tqdm
        abs_path = Path(filepath).resolve()
        print(f"DEBUG: Processing records from directory: {abs_path}")
        
        for record_name in tqdm(record_list, desc="Processing ECG records", unit="record"):
            try:
                record_file = abs_path / f"{record_name}.hea"
                if not record_file.exists():
                    tqdm.write(f"WARNING: File not found: {record_file}")
                    failed_records.append(record_name)
                    continue
                    
                features, _, _ = self.process_record(filepath, record_name)
                features['record_id'] = record_name
                all_features.append(features)
                
            except Exception as e:
                tqdm.write(f"Failed to process record {record_name}: {e}")
                failed_records.append(record_name)
        
        # Create DataFrame
        df = pd.DataFrame(all_features) if all_features else pd.DataFrame()
        
        # Save features if we have any
        if len(all_features) > 0:
            self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.config.PROCESSED_DATA_DIR / 'ecg_features.csv', index=False)
        else:
            print("WARNING: No features were extracted!")
        
        # Save processed data
        processed_data = {
            'features': df,
            'failed_records': failed_records,
            'feature_names': df.columns.tolist() if len(df) > 0 else []
        }
        
        import joblib
        joblib.dump(processed_data, self.config.PROCESSED_DATA_DIR / 'ecg_processed.pkl')
        
        print(f"\n=== ECG Processing Complete ===")
        print(f"Processed {len(all_features)} records")
        print(f"Failed: {len(failed_records)} records")
        if len(df) > 0:
            print(f"Feature shape: {df.shape}")
        
        return processed_data