# src/preprocessing/mri_preprocessor.py
"""
MRI data preprocessing module.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from skimage import exposure, filters, morphology, measure
from skimage.transform import resize
from scipy import ndimage
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MRIPreprocessor:
    """Preprocessor for cardiac MRI (ACDC) dataset."""
    
    def __init__(self, config):
        self.config = config
        self.target_size = config.MRI_TARGET_SIZE
        
    def load_nifti(self, filepath):
        """Load NIfTI MRI file."""
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        return data, affine, header
    
    def normalize_intensity(self, image_data):
        """Normalize MRI intensity."""
        if self.config.MRI_NORMALIZATION == "zscore":
            # Z-score normalization
            mean = np.mean(image_data)
            std = np.std(image_data)
            normalized = (image_data - mean) / (std + 1e-10)
        
        elif self.config.MRI_NORMALIZATION == "minmax":
            # Min-max normalization
            min_val = np.min(image_data)
            max_val = np.max(image_data)
            normalized = (image_data - min_val) / (max_val - min_val + 1e-10)
        
        elif self.config.MRI_NORMALIZATION == "histogram":
            # Histogram equalization
            normalized = exposure.equalize_hist(image_data)
        
        else:
            normalized = image_data
        
        return normalized
    
    def resize_image(self, image_data):
        """Resize MRI image to target size."""
        # If 4D (cine MRI with time dimension), take the first time frame
        if len(image_data.shape) == 4:
            image_data = image_data[:, :, :, 0]
            
        if len(image_data.shape) == 3:
            # 3D volume
            resized = np.zeros((self.target_size[0], self.target_size[1], image_data.shape[2]))
            for i in range(image_data.shape[2]):
                resized[:, :, i] = resize(image_data[:, :, i], self.target_size, 
                                          preserve_range=True, anti_aliasing=True)
        else:
            # 2D slice
            resized = resize(image_data, self.target_size, preserve_range=True, anti_aliasing=True)
        
        return resized
    
    def segment_ventricles(self, image_slice):
        """Segment left and right ventricles from MRI slice."""
        # Apply Gaussian blur for noise reduction
        blurred = ndimage.gaussian_filter(image_slice, sigma=2)
        
        # Adaptive thresholding
        threshold = filters.threshold_otsu(blurred)
        binary = blurred > threshold
        
        # Fill holes
        binary = morphology.remove_small_holes(binary, area_threshold=500)
        binary = morphology.remove_small_objects(binary, min_size=500)
        
        # Label connected components
        labeled = measure.label(binary)
        
        # Get region properties
        regions = measure.regionprops(labeled)
        
        # Identify LV and RV based on area and position
        lv = None
        rv = None
        center_y, center_x = np.array(image_slice.shape) / 2
        
        for region in regions:
            # Calculate distance to center
            cy, cx = region.centroid
            distance = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
            
            if region.area > 500:  # Minimum area threshold
                if distance < 50:  # Likely LV (close to center)
                    lv = region
                elif distance > 50:  # Likely RV (away from center)
                    rv = region
        
        return lv, rv, labeled
    
    def extract_structural_features(self, image_data):
        """Extract structural features from MRI."""
        features = {}
        
        # Use mid-ventricular slice for analysis
        if len(image_data.shape) == 3:
            # For 3D volume, take middle slice
            mid_slice = image_data[:, :, image_data.shape[2] // 2]
        else:
            mid_slice = image_data
        
        # Segment ventricles
        lv, rv, labeled = self.segment_ventricles(mid_slice)
        
        if lv:
            # LV features
            features['lv_area'] = lv.area
            features['lv_perimeter'] = lv.perimeter
            features['lv_eccentricity'] = lv.eccentricity
            features['lv_solidity'] = lv.solidity
            
            # Calculate LV diameter
            lv_minr, lv_minc, lv_maxr, lv_maxc = lv.bbox
            features['lv_diameter'] = max(lv_maxr - lv_minr, lv_maxc - lv_minc)
            
            # LV intensity features
            lv_mask = labeled == lv.label
            features['lv_intensity_mean'] = np.mean(mid_slice[lv_mask])
            features['lv_intensity_std'] = np.std(mid_slice[lv_mask])
        
        if rv:
            # RV features
            features['rv_area'] = rv.area
            features['rv_perimeter'] = rv.perimeter
            features['rv_eccentricity'] = rv.eccentricity
            features['rv_solidity'] = rv.solidity
            
            # RV intensity features
            rv_mask = labeled == rv.label
            features['rv_intensity_mean'] = np.mean(mid_slice[rv_mask])
            features['rv_intensity_std'] = np.std(mid_slice[rv_mask])
        
        # Global features
        features['global_intensity_mean'] = np.mean(image_data)
        features['global_intensity_std'] = np.std(image_data)
        features['global_entropy'] = self._calculate_entropy(image_data)
        
        return features
    
    def extract_functional_features(self, cine_sequence):
        """Extract functional features from cine MRI sequence."""
        features = {}
        
        # Assuming cine_sequence is a 4D array (height, width, time, slices)
        if len(cine_sequence.shape) == 4:
            n_frames = cine_sequence.shape[2]
            n_slices = cine_sequence.shape[3]
            
            lv_areas = []
            rv_areas = []
            
            # Track ventricular areas over time
            for frame in range(n_frames):
                # Use mid-ventricular slice
                slice_idx = n_slices // 2
                current_slice = cine_sequence[:, :, frame, slice_idx]
                
                lv, rv, _ = self.segment_ventricles(current_slice)
                
                if lv:
                    lv_areas.append(lv.area)
                if rv:
                    rv_areas.append(rv.area)
            
            # Calculate functional metrics
            if lv_areas:
                # End-diastolic and end-systolic volumes
                edv = max(lv_areas)  # Maximum area = end-diastole
                esv = min(lv_areas)  # Minimum area = end-systole
                
                # Ejection fraction
                features['lv_ejection_fraction'] = (edv - esv) / edv * 100
                
                # Stroke volume
                features['lv_stroke_volume'] = edv - esv
                
                # Contraction index (normalized change)
                features['lv_contraction_index'] = (edv - esv) / (edv + 1e-10)
                
                # Wall motion score (simplified)
                motion_range = np.ptp(lv_areas)
                features['lv_wall_motion_score'] = motion_range / (edv + 1e-10)
            
            if rv_areas:
                edv_rv = max(rv_areas)
                esv_rv = min(rv_areas)
                features['rv_ejection_fraction'] = (edv_rv - esv_rv) / edv_rv * 100
                features['rv_stroke_volume'] = edv_rv - esv_rv
            
            # Calculate heart rate from cine (if timing info available)
            # This would require DICOM header info
        
        return features
    
    def extract_shape_descriptors(self, image_data):
        """Extract shape descriptors of the heart."""
        features = {}
        
        # Use mid-ventricular slice
        if len(image_data.shape) == 3:
            mid_slice = image_data[:, :, image_data.shape[2] // 2]
        else:
            mid_slice = image_data
        
        # Threshold to get heart region
        threshold = filters.threshold_otsu(mid_slice)
        binary = mid_slice > threshold
        
        # Get largest connected component (heart)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        if regions:
            # Find region with largest area
            heart_region = max(regions, key=lambda x: x.area)
            
            # Basic shape features
            features['heart_area'] = heart_region.area
            features['heart_perimeter'] = heart_region.perimeter
            features['heart_eccentricity'] = heart_region.eccentricity
            features['heart_solidity'] = heart_region.solidity
            features['heart_extent'] = heart_region.extent
            
            # Circularity (4π × area / perimeter²)
            if heart_region.perimeter > 0:
                features['heart_circularity'] = (4 * np.pi * heart_region.area) / (heart_region.perimeter ** 2)
            else:
                features['heart_circularity'] = 0
            
            # Aspect ratio
            minr, minc, maxr, maxc = heart_region.bbox
            features['heart_aspect_ratio'] = (maxr - minr) / (maxc - minc + 1e-10)
            
            # Convex hull features
            convex_area = heart_region.convex_area
            features['heart_convexity'] = heart_region.area / (convex_area + 1e-10)
        
        return features
    
    def _calculate_entropy(self, data):
        """Calculate entropy of image data."""
        # Normalize data
        data_flat = data.flatten()
        hist, _ = np.histogram(data_flat, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def compute_severity_group(self, features):
        """Assign severity group based on extracted features."""
        severity_score = 0
        
        # LV dysfunction indicators
        if features.get('lv_ejection_fraction', 55) < 50:
            severity_score += 2
        elif features.get('lv_ejection_fraction', 55) < 55:
            severity_score += 1
        
        # RV dysfunction indicators
        if features.get('rv_ejection_fraction', 55) < 50:
            severity_score += 2
        elif features.get('rv_ejection_fraction', 55) < 55:
            severity_score += 1
        
        # Wall motion abnormalities
        if features.get('lv_wall_motion_score', 0.2) > 0.3:
            severity_score += 1
        
        # Ventricular enlargement
        if features.get('lv_area', 1000) > 2000:
            severity_score += 1
        if features.get('rv_area', 800) > 1500:
            severity_score += 1
        
        # Assign group
        if severity_score == 0:
            severity_group = 'normal'
        elif severity_score <= 2:
            severity_group = 'remodeling'
        else:
            severity_group = 'dysfunction'
        
        return severity_group, severity_score
    
    def process_case(self, filepath):
        """Process a single MRI case."""
        try:
            # Load MRI data
            image_data, affine, header = self.load_nifti(filepath)
            
            # Normalize intensity
            normalized = self.normalize_intensity(image_data)
            
            # Resize
            resized = self.resize_image(normalized)
            
            # Extract structural features
            structural_features = self.extract_structural_features(resized)
            
            # Extract shape descriptors
            shape_features = self.extract_shape_descriptors(resized)
            
            # Combine features
            all_features = {**structural_features, **shape_features}
            
            # If cine sequence available, extract functional features
            # This would require handling of cine sequences
            
            # Compute severity group
            severity_group, severity_score = self.compute_severity_group(all_features)
            all_features['severity_group'] = severity_group
            all_features['severity_score'] = severity_score
            
            return all_features, resized
        
        except Exception as e:
            print(f"Error processing MRI case: {e}")
            return None, None
    
    def process_all_cases(self, filepath):
        """Process all MRI cases in directory."""
        all_features = []
        failed_cases = []
        
        # Find all NIfTI files
        mri_files = list(Path(filepath).glob("**/*.nii.gz"))
        
        from tqdm import tqdm
        for mri_file in tqdm(mri_files, desc="Processing MRI cases", unit="case"):
            try:
                features, _ = self.process_case(str(mri_file))
                
                if features:
                    features['case_id'] = mri_file.stem
                    all_features.append(features)
                    tqdm.write(f"  ✓ {mri_file.name}")
                else:
                    failed_cases.append(mri_file.name)
                    tqdm.write(f"  ✗ Failed: {mri_file.name}")
                    
            except Exception as e:
                tqdm.write(f"  ✗ Error: {mri_file.name} — {e}")
                failed_cases.append(mri_file.name)

        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Handle missing values by filling with 0
        df = df.fillna(0)
        
        # Save features
        df.to_csv(self.config.PROCESSED_DATA_DIR / 'mri_features.csv', index=False)
        
        # Save processed data
        processed_data = {
            'features': df,
            'failed_cases': failed_cases,
            'feature_names': df.columns.tolist()
        }
        
        import joblib
        joblib.dump(processed_data, self.config.PROCESSED_DATA_DIR / 'mri_processed.pkl')
        
        print(f"\n=== MRI Processing Complete ===")
        print(f"Processed {len(all_features)} cases")
        print(f"Failed: {len(failed_cases)} cases")
        print(f"Feature shape: {df.shape}")
        
        return processed_data