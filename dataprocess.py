import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold

def extract_features(file_path, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        # 1. MFCCs and their derivatives (Deltas)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # 2. Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        
        # 3. Energy Feature
        rms = librosa.feature.rms(y=audio)

        # Combine all frame-level features
        all_features = np.vstack([
            mfccs, delta_mfccs, delta2_mfccs, 
            spectral_centroid, spectral_bandwidth, spectral_rolloff, 
            zero_crossing_rate, rms
        ])

        # 4. Comprehensive Statistical Aggregation across time (axis=1)
        feat_mean = np.mean(all_features, axis=1)
        feat_std = np.std(all_features, axis=1)
        feat_min = np.min(all_features, axis=1)
        feat_max = np.max(all_features, axis=1)
        feat_median = np.median(all_features, axis=1)

        # Concatenate into a single 1D feature vector for this audio file
        features = np.concatenate((
            feat_mean, feat_std, feat_min, feat_max, feat_median
        ))
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_dir):
    """
    AD (Alzheimer's Dementia) -> Label 1
    CN (Control Normal) -> Label 0
    """
    features = []
    labels = []
    
    classes = {'ad': 1, 'cn': 0}
    
    for cls_name, label in classes.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Directory not found: {cls_dir}")
            continue
            
        for file_name in os.listdir(cls_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(cls_dir, file_name)
                feature = extract_features(file_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(label)
                
    return np.array(features), np.array(labels)

def load_or_extract_features(data_dir, cache_path="extracted_features.npz"):
    if os.path.exists(cache_path):
        print(f"Loading features from cache: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']
    else:
        print(f"Cache not found. Extracting features from {data_dir}...")
        X, y = load_data(data_dir)

        print(f"Saving extracted features to {cache_path}...")
        np.savez_compressed(cache_path, X=X, y=y)
        return X, y

def get_kfold_splits(X, y, n_splits=5):
    """
    K-Fold cross validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return skf.split(X, y)

def aggregate_features(frame_level_features):
    """
    Instead of just returning np.mean(frame_level_features, axis=0),
    concatenate multiple statistics to get a richer representation.
    """
    mean_vals = np.mean(frame_level_features, axis=0)
    std_vals = np.std(frame_level_features, axis=0)
    min_vals = np.min(frame_level_features, axis=0)
    max_vals = np.max(frame_level_features, axis=0)
    
    # Optionally add skewness and kurtosis
    skew_vals = scipy.stats.skew(frame_level_features, axis=0)
    kurt_vals = scipy.stats.kurtosis(frame_level_features, axis=0)
    
    # Combine all these statistics into a single 1D feature vector for the audio file
    return np.concatenate([
        mean_vals, std_vals, min_vals, max_vals, skew_vals, kurt_vals
    ])