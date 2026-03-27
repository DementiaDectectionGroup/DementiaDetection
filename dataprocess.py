import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold


def extract_features(file_path):
    """
    Extract a fixed-length acoustic feature vector from a single .wav file.
    Feature groups (can be toggled in Phase 3 for ablation):
      - mfcc        : 40 MFCCs  × 5 stats = 200
      - delta_mfcc  : 40 delta  × 5 stats = 200
      - delta2_mfcc : 40 delta² × 5 stats = 200
      - spectral    : centroid, bandwidth, rolloff × 5 stats = 15
      - zcr         : 1 × 5 stats = 5
      - rms         : 1 × 5 stats = 5
    Total: 625 features
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)

        # ── Feature groups ──────────────────────────────────────────────
        mfccs        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        spectral_centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr                = librosa.feature.zero_crossing_rate(y=audio)
        rms                = librosa.feature.rms(y=audio)

        all_features = np.vstack([
            mfccs, delta_mfccs, delta2_mfccs,
            spectral_centroid, spectral_bandwidth, spectral_rolloff,
            zcr, rms,
        ])

        # ── Statistical aggregation (5 stats per feature row) ───────────
        feat_mean   = np.mean(all_features,   axis=1)
        feat_std    = np.std(all_features,    axis=1)
        feat_min    = np.min(all_features,    axis=1)
        feat_max    = np.max(all_features,    axis=1)
        feat_median = np.median(all_features, axis=1)

        return np.concatenate([feat_mean, feat_std, feat_min, feat_max, feat_median])

    except Exception as e:
        print(f"  [ERROR] Processing {file_path}: {e}")
        return None


def load_data(data_dir):
    """
    Walk data_dir/ad and data_dir/cn, extract features.
    AD → label 1, CN → label 0.
    Files are sorted for reproducibility.
    """
    features, labels, file_ids = [], [], []
    classes = {'ad': 1, 'cn': 0}

    for cls_name, label in classes.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"  [WARN] Directory not found: {cls_dir}")
            continue

        wav_files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.wav'))
        print(f"  Found {len(wav_files)} .wav files in '{cls_name}'")

        for file_name in wav_files:
            file_path = os.path.join(cls_dir, file_name)
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(label)
                file_ids.append(os.path.splitext(file_name)[0])

    return np.array(features), np.array(labels), np.array(file_ids)


def load_or_extract_features(data_dir, cache_path="extracted_features.npz"):
    """
    Load from cache if available; otherwise extract and save.
    Cache stores X, y, and file_ids for traceability.
    """
    if os.path.exists(cache_path):
        print(f"Loading features from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], data['file_ids']

    print(f"Cache not found. Extracting features from: {data_dir}")
    X, y, file_ids = load_data(data_dir)
    print(f"Saving {X.shape[0]} samples to cache: {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y, file_ids=file_ids)
    return X, y, file_ids


def get_kfold_splits(X, y, n_splits=5):
    """Stratified K-Fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y))