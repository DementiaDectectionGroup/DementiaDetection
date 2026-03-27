import os
import numpy as np
import librosa
"""
dataprocess.py – Feature extraction, caching, and CV splits.

Feature groups (Phase 3):
┌──────────────────────┬────────┬──────────────────────────────────────────────┐
│ Group key            │  Dims  │ Description                                  │
├──────────────────────┼────────┼──────────────────────────────────────────────┤
│ mfcc                 │  200   │ 40 MFCCs × 5 stats (mean/std/min/max/median) │
│ delta_mfcc           │  200   │ 40 delta-MFCCs × 5 stats                     │
│ delta2_mfcc          │  200   │ 40 delta²-MFCCs × 5 stats                    │
│ spectral             │   15   │ centroid / bandwidth / rolloff × 5 stats     │
│ energy               │   10   │ ZCR × 5 stats + RMS × 5 stats                │
│ prosodic             │   11   │ F0 stats (6) + voiced/unvoiced ratio (1)      │
│                      │        │ + speech-activity ratio (1) + pause stats (3)│
│ fluency              │    4   │ speech-rate proxy + pause count + pause       │
│                      │        │ duration stats (mean/std)                     │
└──────────────────────┴────────┴──────────────────────────────────────────────┘

Prosodic / fluency approximations
──────────────────────────────────
  • F0  : librosa.yin() on voiced frames only (voiced = RMS > adaptive threshold).
           yin() is a lightweight autocorrelation-based pitch estimator; no
           external dependencies required.
  • Voiced frames : RMS-energy threshold at mean(RMS) * 0.15 (tunable constant).
           This is a reasonable approximation for speech/non-speech segmentation
           without a full VAD model.
  • Pauses : runs of consecutive *unvoiced* frames longer than 0.15 s (3 × 50 ms
           hop = 150 ms) are counted as pauses.  This gives an energy-based
           pause estimate; disfluencies shorter than 150 ms are ignored.
  • Speech rate proxy : number of voiced-frame runs per second of audio.
           Each voiced run roughly corresponds to a syllable nucleus.  This is
           a coarse but well-established proxy (Trevisan et al. 2021).

Cache versioning
────────────────
  The .npz cache stores a 'cache_version' scalar.  If the version in the file
  does not match CACHE_VERSION, the cache is silently ignored and re-extracted.
  Old Phase-1/2 caches (no version key) are also re-extracted automatically.
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold

# ── Cache versioning ──────────────────────────────────────────────────────────
CACHE_VERSION = 3   # increment whenever the feature schema changes

# ── Voiced-frame detection threshold (fraction of mean RMS) ──────────────────
_VOICED_THRESH_FACTOR = 0.15

# ── Minimum pause duration in frames (hop_length = 512 @ 22050 Hz ≈ 23 ms/frame)
_MIN_PAUSE_FRAMES = 7   # ~160 ms  (conservative to avoid counting micro-pauses)

# ── Named feature groups and their ordering ──────────────────────────────────
ALL_GROUPS = ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral', 'energy',
              'prosodic', 'fluency']

# Pre-defined ablation presets  →  used by main.py
FEATURE_PRESETS = {
    'all':                    ALL_GROUPS,
    'baseline_acoustic':      ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral', 'energy'],
    'prosody_only':           ['prosodic'],
    'fluency_only':           ['fluency'],
    'baseline_plus_prosody':  ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral', 'energy', 'prosodic'],
    'baseline_plus_fluency':  ['mfcc', 'delta_mfcc', 'delta2_mfcc', 'spectral', 'energy', 'fluency'],
    'prosody_and_fluency':    ['prosodic', 'fluency'],
}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _five_stats(arr2d: np.ndarray) -> np.ndarray:
    """Return [mean, std, min, max, median] along axis=1, concatenated."""
    return np.concatenate([
        np.mean(arr2d,   axis=1),
        np.std(arr2d,    axis=1),
        np.min(arr2d,    axis=1),
        np.max(arr2d,    axis=1),
        np.median(arr2d, axis=1),
    ])


def _voiced_mask(rms: np.ndarray) -> np.ndarray:
    """
    Boolean mask of voiced frames based on RMS energy threshold.
    rms shape: (1, T)  →  returns (T,) boolean array.
    """
    rms_flat  = rms[0]
    threshold = rms_flat.mean() * _VOICED_THRESH_FACTOR
    return rms_flat > threshold


def _pause_stats(voiced: np.ndarray, sr: int, hop_length: int) -> dict:
    """
    Given a voiced boolean mask, find runs of silence and compute:
      - n_pauses          : number of pauses above minimum duration
      - pause_ratio       : fraction of total frames that are pauses
      - mean_pause_dur_s  : mean pause duration in seconds
      - std_pause_dur_s   : std pause duration in seconds
    """
    unvoiced = ~voiced
    n_frames  = len(voiced)
    hop_s     = hop_length / sr         # seconds per frame

    pauses = []
    i = 0
    while i < n_frames:
        if unvoiced[i]:
            j = i
            while j < n_frames and unvoiced[j]:
                j += 1
            duration_frames = j - i
            if duration_frames >= _MIN_PAUSE_FRAMES:
                pauses.append(duration_frames)
            i = j
        else:
            i += 1

    n_pauses = len(pauses)
    pause_ratio = float(unvoiced.sum()) / n_frames if n_frames > 0 else 0.0

    if n_pauses > 0:
        durations_s     = np.array(pauses) * hop_s
        mean_pause_dur  = float(durations_s.mean())
        std_pause_dur   = float(durations_s.std())
    else:
        mean_pause_dur = 0.0
        std_pause_dur  = 0.0

    return dict(n_pauses=n_pauses, pause_ratio=pause_ratio,
                mean_pause_dur=mean_pause_dur, std_pause_dur=std_pause_dur)


def _speech_rate_proxy(voiced: np.ndarray) -> float:
    """
    Count voiced-segment runs (transitions False→True) per second of audio.
    Each run loosely corresponds to a syllable nucleus.
    Returns voiced runs per second.
    """
    n_frames = len(voiced)
    if n_frames < 2:
        return 0.0
    # count rising edges (unvoiced→voiced transitions)
    runs = int(np.sum(~voiced[:-1] & voiced[1:]))
    total_s = n_frames / max(n_frames, 1)   # relative; normalise by voiced fraction
    voiced_s = voiced.sum()
    if voiced_s == 0:
        return 0.0
    # runs per voiced-second (better than total-second for variable-length clips)
    return runs / (voiced_s * 1.0)


# ── Per-file feature extraction ───────────────────────────────────────────────

def extract_feature_groups(file_path: str) -> dict | None:
    """
    Extract all feature groups from a single .wav file.
    Returns a dict  {group_name: np.ndarray}  or None on failure.

    All arrays are 1-D; concatenation order is defined by ALL_GROUPS.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        hop_length = 512

        # ── Existing acoustic groups ──────────────────────────────────
        mfccs        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        spectral_centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)

        groups = {
            'mfcc':        _five_stats(mfccs),                           # 200
            'delta_mfcc':  _five_stats(delta_mfccs),                     # 200
            'delta2_mfcc': _five_stats(delta2_mfccs),                    # 200
            'spectral':    _five_stats(
                np.vstack([spectral_centroid,
                           spectral_bandwidth,
                           spectral_rolloff])),                           # 15
            'energy':      np.concatenate([_five_stats(zcr),
                                           _five_stats(rms)]),            # 10
        }

        # ── Prosodic group ────────────────────────────────────────────
        voiced = _voiced_mask(rms)                                # (T,) bool
        speech_activity_ratio = float(voiced.sum()) / len(voiced) if len(voiced) > 0 else 0.0

        voiced_audio_frames = np.where(voiced)[0]
        if len(voiced_audio_frames) > 10:
            # yin works on raw audio samples; we need voiced audio segments
            # Approximate: collect all frames marked voiced, compute F0 on full
            # audio but restrict stats to voiced frames only.
            # fmin/fmax chosen for human speech (80–400 Hz).
            f0 = librosa.yin(audio, fmin=80, fmax=400,
                             hop_length=hop_length,
                             frame_length=hop_length * 4)
            # f0 may be shorter than rms by 1–2 frames due to centering; align
            min_len = min(len(f0), len(voiced))
            f0_voiced = f0[:min_len][voiced[:min_len]]
            f0_voiced = f0_voiced[f0_voiced > 0]   # drop zero (unvoiced) estimates
        else:
            f0_voiced = np.array([])

        if len(f0_voiced) > 0:
            f0_mean   = float(f0_voiced.mean())
            f0_std    = float(f0_voiced.std())
            f0_min    = float(f0_voiced.min())
            f0_max    = float(f0_voiced.max())
            f0_median = float(np.median(f0_voiced))
            f0_range  = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_median = f0_range = 0.0

        unvoiced_ratio = 1.0 - speech_activity_ratio
        pause_info     = _pause_stats(voiced, sr, hop_length)

        groups['prosodic'] = np.array([
            f0_mean, f0_std, f0_min, f0_max, f0_median, f0_range,  # 6
            speech_activity_ratio,                                   # 1
            unvoiced_ratio,                                          # 1
            pause_info['pause_ratio'],                               # 1
            pause_info['mean_pause_dur'],                            # 1
            pause_info['std_pause_dur'],                             # 1
        ], dtype=np.float32)                                         # total: 11

        # ── Fluency group ─────────────────────────────────────────────
        speech_rate = _speech_rate_proxy(voiced)
        groups['fluency'] = np.array([
            speech_rate,                    # voiced runs / voiced-second  (1)
            pause_info['n_pauses'],         # count                        (1)
            pause_info['mean_pause_dur'],   # seconds                      (1)
            pause_info['std_pause_dur'],    # seconds                      (1)
        ], dtype=np.float32)               # total: 4

        return groups

    except Exception as e:
        print(f"  [ERROR] Processing {file_path}: {e}")
        return None


def groups_to_vector(groups: dict, selected: list[str]) -> np.ndarray:
    """Concatenate selected group arrays into a single feature vector."""
    return np.concatenate([groups[g] for g in selected])


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Walk data_dir/ad and data_dir/cn, extract features for each file.
    Returns:
        group_arrays : dict {group_name: np.ndarray shape (N, D_group)}
        y            : np.ndarray shape (N,)
        file_ids     : np.ndarray shape (N,)
    """
    all_groups: list[dict] = []
    labels:    list[int]   = []
    file_ids:  list[str]   = []
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
            feat = extract_feature_groups(file_path)
            if feat is not None:
                all_groups.append(feat)
                labels.append(label)
                file_ids.append(os.path.splitext(file_name)[0])

    # Stack each group independently → shape (N, D_group)
    group_arrays = {g: np.stack([s[g] for s in all_groups]) for g in ALL_GROUPS}
    return group_arrays, np.array(labels), np.array(file_ids)


# ── Cache ─────────────────────────────────────────────────────────────────────

def load_or_extract_features(data_dir: str, cache_path: str = "extracted_features.npz"
                             ) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Load per-group feature arrays from cache if valid, else re-extract.

    Cache format (Phase 3):
      - 'cache_version' : scalar int
      - 'y'             : (N,)
      - 'file_ids'      : (N,)
      - 'grp_<name>'    : (N, D) for each group in ALL_GROUPS

    Backward-compatibility: caches without 'cache_version' or with an older
    version are silently ignored and regenerated.
    """
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            stored_version = int(data['cache_version']) if 'cache_version' in data else -1
            if stored_version == CACHE_VERSION:
                print(f"Loading features from cache: {cache_path}  (v{CACHE_VERSION})")
                group_arrays = {g: data[f'grp_{g}'] for g in ALL_GROUPS}
                return group_arrays, data['y'], data['file_ids']
            else:
                print(f"Cache version mismatch (stored={stored_version}, "
                      f"required={CACHE_VERSION}). Re-extracting...")
        except Exception as e:
            print(f"Cache unreadable ({e}). Re-extracting...")

    print(f"Extracting features from: {data_dir}")
    group_arrays, y, file_ids = load_data(data_dir)

    save_dict = {
        'cache_version': np.array(CACHE_VERSION),
        'y':             y,
        'file_ids':      file_ids,
    }
    for g, arr in group_arrays.items():
        save_dict[f'grp_{g}'] = arr

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {len(y)} samples to cache: {cache_path}  (v{CACHE_VERSION})")
    return group_arrays, y, file_ids


def select_features(group_arrays: dict, groups: list[str]) -> np.ndarray:
    """
    Concatenate the requested groups horizontally → (N, D_total).
    Validates that all requested group names exist.
    """
    unknown = [g for g in groups if g not in ALL_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. "
                         f"Valid: {ALL_GROUPS}")
    return np.concatenate([group_arrays[g] for g in groups], axis=1)


def get_kfold_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> list:
    """Stratified K-Fold splits. Unchanged from Phase 1/2."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y))