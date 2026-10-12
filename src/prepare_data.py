"""
prepare_data.py
===============
Data preparation for the "Edge-Ready ECG Classification" project (CRISP-DM,
Stage 3). Uses the MIT-BIH Arrhythmia Database (MITDB) and, optionally, the
MIT-BIH Noise Stress Test Database (NSTDB) for the noise-robustness experiment.

Pipeline (the order matters and is stored in preprocessing_config.json):

    raw ECG (lead MLII)
      -> [optional] add NSTDB noise at a given SNR  (only for noisy TEST sets)
      -> Butterworth band-pass 0.5-40 Hz (zero-phase by default)
      -> beat segmentation around annotated R-peaks (180 before, 179 after)
      -> drop flat-line beats (documented)
      -> per-beat normalisation (min-max to [-1, 1] by default)
      -> model input (float32, shape (N, 360))


Usage:
    python prepare_data.py --data-dir ../data/raw --output-dir ../data/processed
    python prepare_data.py --make-noisy-test

"""

import os
import sys
import json
import argparse
import platform
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import wfdb
from scipy import signal as scipy_signal

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42

# Dataset configuration
SAMPLING_FREQ = 360          # Hz, MIT-BIH sampling frequency

# Beat extraction window: [R - 180, R + 179] -> 360 samples (1.0 s at 360 Hz)
SAMPLES_BEFORE_R = 180
SAMPLES_AFTER_R = 179
INPUT_LENGTH = SAMPLES_BEFORE_R + SAMPLES_AFTER_R + 1
assert INPUT_LENGTH == 360, "Model input length must stay 360 samples"

# Band-pass filter
FILTER_LOWCUT = 0.5          # Hz  (removes baseline wander)
FILTER_HIGHCUT = 40.0        # Hz  (removes EMG / mains noise)
FILTER_ORDER = 4

# Flat-line beat rejection: a beat whose peak-to-peak range (in mV, after
# filtering) is below this value carries no ECG information.
MIN_BEAT_PTP_MV = 1e-3

# Standard inter-patient split (de Chazal et al.)
DS1_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]
DS2_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

# Train / validation split INSIDE DS1 (by record, never by beat)
TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209'
]
VALIDATION_RECORDS = ['215', '220', '223', '230']

# Hard safety checks: a wrong split would silently invalidate every result.
assert set(TRAIN_RECORDS) | set(VALIDATION_RECORDS) == set(DS1_RECORDS)
assert not (set(TRAIN_RECORDS) & set(VALIDATION_RECORDS))
assert not (set(DS1_RECORDS) & set(DS2_RECORDS))
assert len(DS1_RECORDS) == 22 and len(DS2_RECORDS) == 22

# ECG lead: MLII is the standard lead for MIT-BIH beat classification.
# NOTE: it is NOT always channel 0! In records 114 and 232 MLII is channel 1.
TARGET_LEAD = 'MLII'

# ---------------------------------------------------------------------------
# Annotation -> class mapping (AAMI EC57)
# ---------------------------------------------------------------------------
ANNOTATION_MAP = {
    # N: normal-type beats
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S: supraventricular ectopic beats
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V: ventricular ectopic beats
    'V': 'V', 'E': 'V',
    # F: fusion beats
    'F': 'F',
}

SYMBOL_DESCRIPTIONS = {
    'N': 'Normal beat', 'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat', 'e': 'Atrial escape beat',
    'j': 'Nodal (junctional) escape beat',
    'A': 'Atrial premature beat', 'a': 'Aberrated atrial premature beat',
    'J': 'Nodal (junctional) premature beat', 'S': 'Supraventricular premature beat',
    'V': 'Premature ventricular contraction', 'E': 'Ventricular escape beat',
    'F': 'Fusion of ventricular and normal beat',
    '/': 'Paced beat', 'f': 'Fusion of paced and normal beat',
    'Q': 'Unclassifiable beat',
    '+': 'Rhythm change', '~': 'Change in signal quality',
    '|': 'Isolated QRS-like artifact', 'x': 'Non-conducted P-wave',
    '!': 'Ventricular flutter wave', '[': 'Start of ventricular flutter/fibrillation',
    ']': 'End of ventricular flutter/fibrillation', '"': 'Comment annotation',
    's': 'ST change', 'T': 'T-wave change', '*': 'Systole', 'D': 'Diastole',
    '=': 'Measurement annotation', '@': 'Link to external data',
    'p': 'P-wave peak', 't': 'T-wave peak', 'u': 'U-wave peak',
    '(': 'Waveform onset', ')': 'Waveform end',
}

# Beat-type annotations that are deliberately NOT used (paced / unclassifiable).
EXCLUDED_BEAT_SYMBOLS = ['/', 'f', 'Q']

# Annotations that are not heartbeats at all (rhythm / quality / wave marks).
NON_BEAT_SYMBOLS = ['+', '~', '|', 'x', '!', '[', ']', '"', 's', 'T', '*',
                    'D', '=', '@', 'p', 't', 'u', '(', ')']

# Kept for backward compatibility with the old notebook.
EXCLUDED_ANNOTATIONS = EXCLUDED_BEAT_SYMBOLS + NON_BEAT_SYMBOLS

LABEL_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
CLASS_ORDER = ['N', 'S', 'V', 'F']

# Noise Stress Test Database
NSTDB_RECORDS = ['bw', 'em', 'ma']   # baseline wander, electrode motion, muscle artifact
DEFAULT_SNR_DB = [12, 6, 0]


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def log_step(step: str):
    print(f"\n>> {step}")


def log_info(info: str):
    print(f"   [INFO] {info}")


def log_warning(msg: str):
    print(f"   [WARNING] {msg}")


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def _missing_records(data_dir: str, records: List[str], exts=('dat', 'hea')) -> List[str]:
    missing = []
    for rec in records:
        if not all(os.path.exists(os.path.join(data_dir, f"{rec}.{e}")) for e in exts):
            missing.append(rec)
    return missing


def download_mitbih_data(data_dir: str, records: List[str]) -> bool:
    """
    Download MIT-BIH Arrhythmia Database records (.dat, .hea, .atr).

    Uses wfdb.dl_database
    """
    log_step("Downloading MIT-BIH Arrhythmia Database from PhysioNet")
    os.makedirs(data_dir, exist_ok=True)

    missing = _missing_records(data_dir, records, exts=('dat', 'hea', 'atr'))
    if not missing:
        log_info("All requested records already on disk")
        return True
    try:
        wfdb.dl_database('mitdb', dl_dir=data_dir, records=missing, annotators=['atr'])
        log_info(f"Downloaded {len(missing)} records")
    except Exception as e:  # network problems, wrong names, ...
        log_warning(f"Download failed: {e}")
        return False
    return not _missing_records(data_dir, records, exts=('dat', 'hea', 'atr'))


def download_nstdb(noise_dir: str) -> bool:
    """Download the noise records bw, em, ma of the MIT-BIH Noise Stress Test DB."""
    log_step("Downloading MIT-BIH Noise Stress Test Database (bw, em, ma)")
    os.makedirs(noise_dir, exist_ok=True)
    if not _missing_records(noise_dir, NSTDB_RECORDS):
        log_info("Noise records already on disk")
        return True
    try:
        wfdb.dl_database('nstdb', dl_dir=noise_dir, records=NSTDB_RECORDS, annotators=None)
    except Exception as e:
        log_warning(f"NSTDB download failed: {e}")
        return False
    return not _missing_records(noise_dir, NSTDB_RECORDS)


# ============================================================================
# SIGNAL PREPROCESSING
# ============================================================================

def design_bandpass_sos(lowcut: float = FILTER_LOWCUT, highcut: float = FILTER_HIGHCUT,
                        fs: float = SAMPLING_FREQ, order: int = FILTER_ORDER) -> np.ndarray:
    """
    Butterworth band-pass in second-order-sections (SOS) form.

    SOS is numerically stable for a very low cut-off like 0.5 Hz at 360 Hz
    (0.5 / 180 = 0.0028). The old (b, a) form can become inaccurate there.
    """
    nyq = fs / 2.0
    return scipy_signal.butter(order, [lowcut / nyq, highcut / nyq],
                               btype='band', output='sos')


def apply_bandpass_filter(ecg_signal: np.ndarray, fs: float = SAMPLING_FREQ,
                          zero_phase: bool = True) -> np.ndarray:
    """
    Band-pass filter an ECG record.

    zero_phase=True  -> sosfiltfilt: no phase distortion (offline analysis).
    zero_phase=False -> sosfilt:     causal, what a real-time device would do.
    The report must state which one is used (see preprocessing_config.json).
    """
    sos = design_bandpass_sos(fs=fs)
    if zero_phase:
        return scipy_signal.sosfiltfilt(sos, ecg_signal)
    return scipy_signal.sosfilt(sos, ecg_signal)


def normalize_signal(ecg_segment: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalise ONE beat segment.

    'minmax': scale to [-1, 1]   (maps naturally to signed INT8 on MAX78002)
    'zscore': zero mean, unit std
    'none'  : unchanged
    """
    if method == 'none':
        return ecg_segment
    if method == 'minmax':
        lo, hi = np.min(ecg_segment), np.max(ecg_segment)
        if hi - lo < 1e-10:
            return np.zeros_like(ecg_segment)
        return 2.0 * (ecg_segment - lo) / (hi - lo) - 1.0
    if method == 'zscore':
        std = np.std(ecg_segment)
        if std < 1e-10:
            return np.zeros_like(ecg_segment)
        return (ecg_segment - np.mean(ecg_segment)) / std
    raise ValueError(f"Unknown normalization method: {method}")


def clip_signal(ecg_signal: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    OPTIONAL record-level clipping at mean +- threshold * std.

    Disabled by default: an ECG is mostly flat with sparse tall R-peaks, so the
    record std is small and a 5-sigma limit can cut off the R-peaks themselves
    (the most informative part). Use --clip-sigma only after checking in the
    EDA notebook how many samples it would change.
    """
    mean, std = np.mean(ecg_signal), np.std(ecg_signal)
    if std < 1e-10:
        return ecg_signal
    return np.clip(ecg_signal, mean - threshold * std, mean + threshold * std)


# ============================================================================
# NOISE (NSTDB) UTILITIES  -- used ONLY for noisy TEST sets / augmentation
# ============================================================================

def load_noise_record(noise_dir: str, name: str) -> np.ndarray:
    """Load channel 0 of an NSTDB noise record (bw, em or ma) as float array."""
    rec = wfdb.rdrecord(os.path.join(noise_dir, name))
    return rec.p_signal[:, 0].astype(np.float64)


def split_noise_train_test(noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a noise record in time: first half -> training augmentation,
    second half -> test-time noise. Train and test therefore never see the
    same noise samples (prevents leakage through the noise source).
    """
    mid = len(noise) // 2
    return noise[:mid], noise[mid:]


def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float,
                     rng: np.random.Generator,
                     fs: float = SAMPLING_FREQ) -> np.ndarray:
    """
    Add noise to a RAW ECG record at a requested SNR (in dB).

    SNR = 10*log10(P_signal / P_noise)
      P_signal = variance of the band-pass filtered clean ECG (ECG content only,
                 without DC offset or baseline drift)
      P_noise  = variance of the noise segment (mean removed) before scaling

    The noise is tiled if shorter than the ECG and started at a random offset.
    Noise is added BEFORE filtering, so the filter removes what it can, exactly
    as it would on a real device.
    """
    n = len(clean)
    start = int(rng.integers(0, len(noise)))
    idx = (start + np.arange(n)) % len(noise)
    seg = noise[idx] - np.mean(noise[idx])

    p_signal = np.var(apply_bandpass_filter(clean, fs=fs))
    p_noise = np.var(seg)
    if p_noise < 1e-12 or p_signal < 1e-12:
        return clean.copy()
    target_p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    return clean + seg * np.sqrt(target_p_noise / p_noise)


# ============================================================================
# BEAT EXTRACTION / CHANNEL SELECTION
# ============================================================================

def extract_beat_segment(ecg_signal: np.ndarray, r_peak_idx: int,
                         samples_before: int = SAMPLES_BEFORE_R,
                         samples_after: int = SAMPLES_AFTER_R) -> Optional[np.ndarray]:
    """Fixed-length segment [R - before, R + after]; None if it leaves the record."""
    start = r_peak_idx - samples_before
    end = r_peak_idx + samples_after + 1
    if start < 0 or end > len(ecg_signal):
        return None
    return ecg_signal[start:end]


def select_ecg_channel(record, target_lead: str = TARGET_LEAD) -> Tuple[np.ndarray, str, bool]:
    """
    Choose the ECG channel BY NAME (not by position).

    Returns (signal, lead_name, fallback_used). If the target lead does not
    exist in the record, channel 0 is used and fallback_used=True so the
    record can be reported in the README / report.
    """
    names = list(record.sig_name)
    if target_lead in names:
        idx = names.index(target_lead)
        return record.p_signal[:, idx], names[idx], False
    return record.p_signal[:, 0], names[0], True


# ============================================================================
# ANNOTATION PROCESSING
# ============================================================================

def process_annotations(annotations) -> Tuple[List[Tuple[int, str, str]], Counter, Counter]:
    """
    Map WFDB annotations to N/S/V/F.

    Returns:
        beats      : list of (sample_index, original_symbol, class)
        excluded   : Counter of symbols deliberately left out (paced, non-beat, ...)
        unmapped   : Counter of symbols that are in NEITHER list (never silent!)
    """
    beats, excluded, unmapped = [], Counter(), Counter()
    for sample_idx, symbol in zip(annotations.sample, annotations.symbol):
        if symbol in ANNOTATION_MAP:
            beats.append((int(sample_idx), symbol, ANNOTATION_MAP[symbol]))
        elif symbol in EXCLUDED_BEAT_SYMBOLS or symbol in NON_BEAT_SYMBOLS:
            excluded[symbol] += 1
        else:
            unmapped[symbol] += 1
    return beats, excluded, unmapped


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_record(record_id: str, data_dir: str,
                          apply_filter: bool = True,
                          normalization: str = 'minmax',
                          clip_sigma: Optional[float] = None,
                          zero_phase: bool = True,
                          noise: Optional[np.ndarray] = None,
                          snr_db: Optional[float] = None,
                          rng: Optional[np.random.Generator] = None,
                          only_r_peaks: Optional[set] = None) -> Optional[Dict]:
    """
    Load one MIT-BIH record and turn it into fixed-size, normalised beats.

    noise / snr_db : if both are given, NSTDB noise is added to the raw signal
                     (used only for noisy TEST sets).
    only_r_peaks   : if given, only these R-peak positions are extracted. Used
                     so a noisy test set contains EXACTLY the beats of the clean
                     test set, in the same order.
    """
    record_path = os.path.join(data_dir, record_id)
    try:
        record = wfdb.rdrecord(record_path)
        annotations = wfdb.rdann(record_path, 'atr')
    except Exception as e:
        log_warning(f"Record {record_id}: failed to load - {e}")
        return None

    fs = record.fs
    if int(fs) != SAMPLING_FREQ:
        log_warning(f"Record {record_id}: fs={fs} Hz, expected {SAMPLING_FREQ} Hz")

    ecg, lead_name, fallback = select_ecg_channel(record)
    n_nan = int(np.isnan(ecg).sum())
    if n_nan:
        log_warning(f"Record {record_id}: {n_nan} NaN samples replaced by 0")
        ecg = np.nan_to_num(ecg)

    # 0) optional noise injection on the RAW signal
    if noise is not None and snr_db is not None:
        ecg = add_noise_at_snr(ecg, noise, snr_db, rng or np.random.default_rng(RANDOM_SEED), fs=fs)

    # 1) filtering  2) optional clipping
    sig = ecg.copy()
    if apply_filter:
        sig = apply_bandpass_filter(sig, fs=fs, zero_phase=zero_phase)
    if clip_sigma is not None:
        sig = clip_signal(sig, threshold=clip_sigma)

    beat_annotations, excluded, unmapped = process_annotations(annotations)
    if unmapped:
        log_warning(f"Record {record_id}: unmapped symbols {dict(unmapped)} (not used)")

    beats, labels, symbols, r_peaks = [], [], [], []
    dropped_boundary, dropped_flat = Counter(), Counter()

    for sample_idx, symbol, cls in beat_annotations:
        if only_r_peaks is not None and sample_idx not in only_r_peaks:
            continue
        # 3) segmentation
        seg = extract_beat_segment(sig, sample_idx)
        if seg is None:
            dropped_boundary[cls] += 1
            continue
        # 4) reject flat-line beats
        if np.ptp(seg) < MIN_BEAT_PTP_MV:
            dropped_flat[cls] += 1
            continue
        # 5) per-beat normalisation
        beats.append(normalize_signal(seg, method=normalization))
        labels.append(cls)
        symbols.append(symbol)
        r_peaks.append(sample_idx)

    return {
        'record_id': record_id,
        'channel': lead_name,
        'lead_fallback': bool(fallback),
        'all_lead_names': list(record.sig_name),
        'fs': fs,
        'n_samples': int(record.sig_len),
        'beats': np.asarray(beats, dtype=np.float32) if beats else np.empty((0, INPUT_LENGTH), np.float32),
        'labels': np.asarray(labels, dtype='<U1'),
        'original_symbols': np.asarray(symbols, dtype='<U1'),
        'r_peak_indices': np.asarray(r_peaks, dtype=np.int64),
        'num_beats': len(beats),
        'annotation_counts': dict(Counter(annotations.symbol)),   # every symbol in the .atr file
        'excluded_counts': dict(excluded),
        'unmapped_counts': dict(unmapped),
        'dropped_boundary': dict(dropped_boundary),
        'dropped_flat': dict(dropped_flat),
    }


def process_all_records(data_dir: str, records: List[str], **kwargs) -> Dict[str, Dict]:
    results = {}
    for record_id in records:
        res = process_single_record(record_id, data_dir, **kwargs)
        if res is None:
            log_warning(f"Record {record_id}: processing failed")
            continue
        results[record_id] = res
        note = f" (lead fallback -> {res['channel']})" if res['lead_fallback'] else ""
        log_info(f"Record {record_id}: lead={res['channel']:<4} beats={res['num_beats']:5d}{note}")
    return results


# ============================================================================
# DATASET ASSEMBLY
# ============================================================================

def assemble_dataset(processed_records: Dict[str, Dict],
                     record_list: List[str]) -> Dict[str, np.ndarray]:
    """Concatenate beats of the given records (order = record_list order)."""
    parts = [processed_records[r] for r in record_list
             if r in processed_records and processed_records[r]['num_beats'] > 0]
    if not parts:
        return {'X': np.empty((0, INPUT_LENGTH), np.float32), 'y': np.array([], dtype='<U1'),
                'record_ids': np.array([], dtype='<U3'), 'symbols': np.array([], dtype='<U1'),
                'r_peaks': np.array([], dtype=np.int64)}
    return {
        'X': np.vstack([p['beats'] for p in parts]).astype(np.float32),
        'y': np.concatenate([p['labels'] for p in parts]),
        'record_ids': np.concatenate([np.full(p['num_beats'], p['record_id'], dtype='<U3') for p in parts]),
        'symbols': np.concatenate([p['original_symbols'] for p in parts]),
        'r_peaks': np.concatenate([p['r_peak_indices'] for p in parts]),
    }


def encode_labels(y_str: np.ndarray) -> np.ndarray:
    """'N','S','V','F' -> 0,1,2,3 (int64)."""
    return np.array([LABEL_MAP[l] for l in y_str], dtype=np.int64)


def compute_class_weights(y: np.ndarray, num_classes: int = 4) -> Dict[int, float]:
    """weight_c = total / (num_classes * count_c). Missing class -> weight 0.0."""
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    return {c: (float(total / (num_classes * counts[c])) if counts[c] > 0 else 0.0)
            for c in range(num_classes)}


# ============================================================================
# SAVE / LOAD
# ============================================================================

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_processed_data(output_dir: str, data_dict: Dict):
    """
    Save arrays as .npy and everything else (dict/list/str/int/float) as .json.
    (The old version silently skipped plain ints/floats and crashed on numpy
    integer dict keys.)
    """
    os.makedirs(output_dir, exist_ok=True)
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            np.save(os.path.join(output_dir, f"{key}.npy"), value)
        else:
            with open(os.path.join(output_dir, f"{key}.json"), 'w') as f:
                json.dump(value, f, indent=2, default=_json_default)
    log_info(f"Data saved to {output_dir}")


def load_processed_data(output_dir: str) -> Dict:
    """Load every .npy / .json file of a processed-data folder into a dict."""
    data = {}
    for filename in sorted(os.listdir(output_dir)):
        name, ext = os.path.splitext(filename)
        path = os.path.join(output_dir, filename)
        if ext == '.npy':
            data[name] = np.load(path, allow_pickle=False)
        elif ext == '.json':
            with open(path) as f:
                data[name] = json.load(f)
    return data


# ============================================================================
# STATISTICS / DOCUMENTATION TABLES
# ============================================================================

def generate_statistics(processed_records: Dict[str, Dict], train_records: List[str],
                        val_records: List[str], test_records: List[str]) -> pd.DataFrame:
    """Table 3.8: Split | #records | record ids | N | S | V | F | Total."""
    rows = []
    for split_name, rec_list in [('Train', train_records), ('Validation', val_records),
                                 ('Test', test_records)]:
        ds = assemble_dataset(processed_records, rec_list)
        cnt = Counter(ds['y'].tolist())
        rows.append({'Split': split_name, 'Records': len(rec_list),
                     'Record IDs': ', '.join(rec_list),
                     'N': cnt.get('N', 0), 'S': cnt.get('S', 0),
                     'V': cnt.get('V', 0), 'F': cnt.get('F', 0),
                     'Total': len(ds['y'])})
    return pd.DataFrame(rows)


def generate_per_record_statistics(processed_records: Dict[str, Dict]) -> pd.DataFrame:
    """One row per record: split, lead used, class counts, everything dropped."""
    split_of = {**{r: 'Train' for r in TRAIN_RECORDS},
                **{r: 'Validation' for r in VALIDATION_RECORDS},
                **{r: 'Test' for r in DS2_RECORDS}}
    rows = []
    for rid, res in processed_records.items():
        cnt = Counter(res['labels'].tolist())
        rows.append({
            'record': rid, 'split': split_of.get(rid, '?'),
            'leads_in_record': '/'.join(res['all_lead_names']),
            'lead_used': res['channel'], 'lead_fallback': res['lead_fallback'],
            'duration_min': round(res['n_samples'] / res['fs'] / 60, 2),
            'N': cnt.get('N', 0), 'S': cnt.get('S', 0), 'V': cnt.get('V', 0), 'F': cnt.get('F', 0),
            'kept_total': res['num_beats'],
            'dropped_boundary': sum(res['dropped_boundary'].values()),
            'dropped_flat': sum(res['dropped_flat'].values()),
            'excluded_paced_or_Q': sum(res['excluded_counts'].get(s, 0) for s in EXCLUDED_BEAT_SYMBOLS),
            'excluded_non_beat': sum(res['excluded_counts'].get(s, 0) for s in NON_BEAT_SYMBOLS),
            'unmapped': sum(res['unmapped_counts'].values()),
        })
    return pd.DataFrame(rows)


def generate_label_mapping_table(processed_records: Dict[str, Dict]) -> pd.DataFrame:
    """
    Table 2.2 + exclusion documentation.
    For every annotation symbol found in the used records: final class (or
    EXCLUDED), description, number of annotations in the files, and the number
    that finally became a beat in the dataset.
    """
    total = Counter()
    kept = Counter()
    for res in processed_records.values():
        total.update(res['annotation_counts'])
        kept.update(res['original_symbols'].tolist())

    rows = []
    for sym, n in sorted(total.items(), key=lambda kv: -kv[1]):
        if sym in ANNOTATION_MAP:
            final, reason = ANNOTATION_MAP[sym], 'used'
        elif sym in EXCLUDED_BEAT_SYMBOLS:
            final, reason = 'EXCLUDED', 'paced / unclassifiable beat (not one of N,S,V,F)'
        elif sym in NON_BEAT_SYMBOLS:
            final, reason = 'EXCLUDED', 'not a heartbeat (rhythm / quality / wave annotation)'
        else:
            final, reason = 'EXCLUDED', 'unknown symbol'
        rows.append({'final_class': final, 'symbol': sym,
                     'description': SYMBOL_DESCRIPTIONS.get(sym, '?'),
                     'n_annotations': n, 'n_in_dataset': kept.get(sym, 0),
                     'reason': reason})
    df = pd.DataFrame(rows)
    order = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'EXCLUDED': 4}
    df['_o'] = df['final_class'].map(order)
    return df.sort_values(['_o', 'n_annotations'], ascending=[True, False]).drop(columns='_o').reset_index(drop=True)


def software_versions() -> Dict[str, str]:
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'pandas': pd.__version__, 'wfdb': wfdb.__version__}


# ============================================================================
# NOISY TEST SETS (for the later robustness experiment)
# ============================================================================

def build_noisy_test_sets(data_dir: str, noise_dir: str, clean_records: Dict[str, Dict],
                          noise_types: List[str], snr_list: List[float],
                          seed: int, **proc_kwargs) -> Dict[str, np.ndarray]:
    """
    Build noisy versions of DS2 using the SECOND HALF of each NSTDB noise record
    (the first half is reserved for training augmentation). Each noisy set holds
    exactly the same beats, in the same order, as the clean test set.
    """
    out = {}
    for noise_type in noise_types:
        _, noise_test = split_noise_train_test(load_noise_record(noise_dir, noise_type))
        for snr in snr_list:
            parts = []
            for i, rid in enumerate(DS2_RECORDS):
                if rid not in clean_records or clean_records[rid]['num_beats'] == 0:
                    continue
                rng = np.random.default_rng(seed + 1000 * i + int(snr) + 7 * len(noise_type))
                res = process_single_record(
                    rid, data_dir, noise=noise_test, snr_db=snr, rng=rng,
                    only_r_peaks=set(clean_records[rid]['r_peak_indices'].tolist()),
                    **proc_kwargs)
                # flat check could in theory differ; assert alignment
                assert res['num_beats'] == clean_records[rid]['num_beats'], \
                    f"Noisy/clean beat mismatch in record {rid}"
                parts.append(res['beats'])
            out[f"X_test_{noise_type}_{int(snr)}dB"] = np.vstack(parts).astype(np.float32)
            log_info(f"Noisy test set {noise_type} @ {snr} dB: {out[f'X_test_{noise_type}_{int(snr)}dB'].shape}")
    return out


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare MIT-BIH data for ECG classification')
    parser.add_argument('--data-dir', type=str, default='../data/raw')
    parser.add_argument('--output-dir', type=str, default='../data/processed')
    parser.add_argument('--noise-dir', type=str, default='../data/nstdb')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--normalization', choices=['minmax', 'zscore', 'none'], default='minmax')
    parser.add_argument('--clip-sigma', type=float, default=None,
                        help='optional record-level clipping at mean +- k*std (off by default)')
    parser.add_argument('--causal', action='store_true',
                        help='use a causal (real-time) filter instead of zero-phase filtfilt')
    parser.add_argument('--make-noisy-test', action='store_true',
                        help='also create noisy DS2 sets from NSTDB (needs ~70 MB per set)')
    parser.add_argument('--noise-types', nargs='+', default=NSTDB_RECORDS)
    parser.add_argument('--snr-db', nargs='+', type=float, default=DEFAULT_SNR_DB)
    args = parser.parse_args()

    np.random.seed(args.seed)
    zero_phase = not args.causal
    proc_kwargs = dict(normalization=args.normalization, clip_sigma=args.clip_sigma,
                       zero_phase=zero_phase)

    log_section("ECG Data Preparation Pipeline")
    log_info(f"Data dir: {args.data_dir} | Output dir: {args.output_dir}")
    log_info(f"Seed: {args.seed} | fs: {SAMPLING_FREQ} Hz | input length: {INPUT_LENGTH} samples")
    log_info(f"Filter: {FILTER_LOWCUT}-{FILTER_HIGHCUT} Hz, order {FILTER_ORDER}, "
             f"{'zero-phase' if zero_phase else 'causal'} | normalisation: {args.normalization}")

    all_records = DS1_RECORDS + DS2_RECORDS

    # 1) download
    if not args.skip_download:
        if not download_mitbih_data(args.data_dir, all_records):
            log_warning("Some records are missing; continuing with what is available")

    # 2) process
    log_section("Processing Records")
    processed = process_all_records(args.data_dir, all_records, **proc_kwargs)
    missing = [r for r in all_records if r not in processed]
    if missing:
        log_warning(f"Records NOT processed: {missing}")

    # 3) assemble per split (always by record)
    log_section("Assembling Datasets")
    train = assemble_dataset(processed, TRAIN_RECORDS)
    val = assemble_dataset(processed, VALIDATION_RECORDS)
    test = assemble_dataset(processed, DS2_RECORDS)
    for name, ds in [('Training', train), ('Validation', val), ('Test', test)]:
        log_info(f"{name}: {len(ds['y'])} beats")

    # no record may appear in two splits (checked on the real beat arrays)
    s_tr, s_va, s_te = (set(d['record_ids'].tolist()) for d in (train, val, test))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "Record leakage between splits!"

    y_train, y_val, y_test = (encode_labels(d['y']) for d in (train, val, test))

    # warn if validation is too small for minority classes
    for cls in CLASS_ORDER:
        n_val = int((val['y'] == cls).sum())
        if n_val < 50:
            log_warning(f"Validation set has only {n_val} '{cls}' beats - "
                        f"metrics for this class will be very noisy (see README)")

    # 4) class weights from TRAIN only
    class_weights = compute_class_weights(y_train)
    log_info(f"Class weights (train): {class_weights}")

    # 5) save
    log_section("Saving Processed Data")
    data_to_save = {
        'X_train': train['X'], 'y_train': y_train,
        'X_val': val['X'], 'y_val': y_val,
        'X_test': test['X'], 'y_test': y_test,
        'train_record_ids': train['record_ids'], 'val_record_ids': val['record_ids'],
        'test_record_ids': test['record_ids'],
        'train_symbols': train['symbols'], 'val_symbols': val['symbols'], 'test_symbols': test['symbols'],
        'train_r_peaks': train['r_peaks'], 'val_r_peaks': val['r_peaks'], 'test_r_peaks': test['r_peaks'],
        'class_weights': class_weights,
        'label_map': LABEL_MAP,
        'split_records': {'train': TRAIN_RECORDS, 'val': VALIDATION_RECORDS, 'test': DS2_RECORDS},
    }
    save_processed_data(args.output_dir, data_to_save)

    # 6) statistics tables
    log_section("Dataset Statistics")
    stats_df = generate_statistics(processed, TRAIN_RECORDS, VALIDATION_RECORDS, DS2_RECORDS)
    print("\n" + stats_df.drop(columns='Record IDs').to_string(index=False))
    stats_df.to_csv(os.path.join(args.output_dir, 'dataset_statistics.csv'), index=False)

    per_rec = generate_per_record_statistics(processed)
    per_rec.to_csv(os.path.join(args.output_dir, 'per_record_statistics.csv'), index=False)
    fallback_recs = per_rec.loc[per_rec['lead_fallback'], 'record'].tolist()
    log_info(f"Records where {TARGET_LEAD} was not available (fallback used): {fallback_recs or 'none'}")
    non_first = per_rec.loc[per_rec['leads_in_record'].str.split('/').str[0] != TARGET_LEAD, 'record'].tolist()
    log_info(f"Records where {TARGET_LEAD} is NOT channel 0 (handled by name): {non_first or 'none'}")

    mapping_df = generate_label_mapping_table(processed)
    print("\n" + mapping_df.to_string(index=False))
    mapping_df.to_csv(os.path.join(args.output_dir, 'label_mapping_counts.csv'), index=False)

    # 7) config (everything needed to reproduce)
    config = {
        'random_seed': args.seed,
        'software_versions': software_versions(),
        'sampling_frequency': SAMPLING_FREQ,
        'input_length': INPUT_LENGTH,
        'samples_before_r': SAMPLES_BEFORE_R,
        'samples_after_r': SAMPLES_AFTER_R,
        'target_lead': TARGET_LEAD,
        'pipeline_order': ['(noise injection, noisy test only)', 'band-pass filter',
                           'optional clipping', 'segmentation', 'flat-beat rejection',
                           'per-beat normalisation'],
        'filter': {'type': 'Butterworth band-pass (SOS)', 'lowcut_hz': FILTER_LOWCUT,
                   'highcut_hz': FILTER_HIGHCUT, 'order': FILTER_ORDER,
                   'zero_phase': zero_phase},
        'normalization_method': args.normalization,
        'clip_sigma': args.clip_sigma,
        'min_beat_ptp_mv': MIN_BEAT_PTP_MV,
        'annotation_map': ANNOTATION_MAP,
        'excluded_beat_symbols': EXCLUDED_BEAT_SYMBOLS,
        'non_beat_symbols': NON_BEAT_SYMBOLS,
        'label_map': LABEL_MAP,
        'ds1_records': DS1_RECORDS, 'ds2_records': DS2_RECORDS,
        'train_records': TRAIN_RECORDS, 'val_records': VALIDATION_RECORDS,
        'test_records': DS2_RECORDS,
        'noise_protocol': {
            'database': 'MIT-BIH Noise Stress Test Database (bw, em, ma)',
            'train_noise': 'first half of each noise record (augmentation only)',
            'test_noise': 'second half of each noise record (DS2 robustness test only)',
            'snr_definition': '10*log10(var(band-pass filtered clean ECG) / var(noise segment))',
            'noise_added_before_filtering': True,
        },
    }
    with open(os.path.join(args.output_dir, 'preprocessing_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=_json_default)
    log_info("Configuration saved")

    # 8) optional noisy test sets
    if args.make_noisy_test:
        log_section("Noisy Test Sets (DS2 only)")
        if not args.skip_download:
            download_nstdb(args.noise_dir)
        noisy = build_noisy_test_sets(args.data_dir, args.noise_dir, processed,
                                      args.noise_types, args.snr_db, args.seed, **proc_kwargs)
        save_processed_data(args.output_dir, noisy)

    log_section("Data Preparation Complete")
    print(f"""
Training beats:   {len(y_train):7d}   ({len(TRAIN_RECORDS)} records)
Validation beats: {len(y_val):7d}   ({len(VALIDATION_RECORDS)} records)
Test beats:       {len(y_test):7d}   ({len(DS2_RECORDS)} records)
Output directory: {args.output_dir}
""")


if __name__ == '__main__':
    main()
