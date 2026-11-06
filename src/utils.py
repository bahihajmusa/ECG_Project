"""
utils.py
========
Small helpers shared by train.py, evaluate.py, noise_test.py, baseline.py and
error_analysis.py: seeding, config loading, metrics, plots, model prediction.

Everything that decides "what is a metric" lives here, so all scripts report
exactly the same numbers.
"""

import os
import json
import random
from typing import Dict, List, Optional

import numpy as np
import yaml

CLASS_NAMES = ['N', 'S', 'V', 'F']
NUM_CLASSES = 4


# ----------------------------------------------------------------------------
# Reproducibility, config, device
# ----------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str) -> Dict:
    """Read a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: Dict, new: Dict) -> Dict:
    """Recursively copy values of `new` into `base` (used for sweeps / overrides)."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_device(name: str = 'auto'):
    """'auto' -> cuda, then mps (Apple), then cpu."""
    import torch
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ----------------------------------------------------------------------------
# Data loading (arrays written by prepare_data.py)
# ----------------------------------------------------------------------------

def load_split(processed_dir: str, split: str):
    """Return (X, y) for split in {'train', 'val', 'test'}."""
    X = np.load(os.path.join(processed_dir, f'X_{split}.npy'))
    y = np.load(os.path.join(processed_dir, f'y_{split}.npy'))
    return X.astype(np.float32), y.astype(np.int64)


def load_meta(processed_dir: str, split: str):
    """Per-beat record ids, original symbols and R-peak positions of a split."""
    out = {}
    for key in ('record_ids', 'symbols', 'r_peaks'):
        p = os.path.join(processed_dir, f'{split}_{key}.npy')
        out[key] = np.load(p) if os.path.exists(p) else None
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Accuracy, macro-F1, per-class precision / recall / F1 and the confusion
    matrix (rows = true class, columns = predicted class).
    A class with no predictions gets precision 0 (zero_division=0), so it is
    never silently ignored.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    labels = list(range(NUM_CLASSES))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(),
        'support': support.tolist(),
        'confusion': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'n': int(len(y_true)),
    }


def metrics_rows(model_name: str, split: str, condition: str, m: Dict,
                 params: Optional[int] = None) -> List[Dict]:
    """Flatten a metrics dict into rows for metrics.csv (one overall row + one per class)."""
    rows = [{'model': model_name, 'split': split, 'condition': condition, 'class': 'overall',
             'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'], 'precision': '', 'recall': '', 'f1': '',
             'support': m['n'], 'parameters': params if params is not None else ''}]
    for c, name in enumerate(CLASS_NAMES):
        rows.append({'model': model_name, 'split': split, 'condition': condition, 'class': name,
                     'accuracy': '', 'macro_f1': '', 'precision': m['precision'][c], 'recall': m['recall'][c],
                     'f1': m['f1'][c], 'support': m['support'][c], 'parameters': ''})
    return rows


def append_rows_csv(path: str, rows: List[Dict]):
    """Append rows to a csv (creates it with a header if needed)."""
    import pandas as pd
    ensure_dir(os.path.dirname(path) or '.')
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def format_metrics_line(m: Dict) -> str:
    f1 = ' '.join(f'{n}:{v:.3f}' for n, v in zip(CLASS_NAMES, m['f1']))
    return f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} | F1 {f1}"


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_confusion_matrix(cm, path: str, title: str = 'Confusion matrix'):
    """Save counts and row-normalised (recall) confusion matrix side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, sub in [(axes[0], cm, '{:.0f}', 'counts'),
                               (axes[1], norm, '{:.2f}', 'row-normalised (recall)')]:
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{title} - {sub}', fontsize=10)
        thr = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, fmt.format(data[i, j]), ha='center', va='center',
                        color='white' if data[i, j] > thr else 'black', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or '.')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_csv(cm, path: str):
    import pandas as pd
    ensure_dir(os.path.dirname(path) or '.')
    pd.DataFrame(cm, index=[f'true_{c}' for c in CLASS_NAMES],
                 columns=[f'pred_{c}' for c in CLASS_NAMES]).to_csv(path)


# ----------------------------------------------------------------------------
# Prediction with a torch model
# ----------------------------------------------------------------------------

def predict(model, X: np.ndarray, device, batch_size: int = 1024, return_probs: bool = False):
    """Run a model over beats X (N, 360); return predicted classes (and softmax probabilities)."""
    import torch
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).float().unsqueeze(1).to(device)
            logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            if return_probs:
                probs.append(torch.softmax(logits, 1).cpu().numpy())
    preds = np.concatenate(preds)
    if return_probs:
        return preds, np.concatenate(probs)
    return preds


def upsert_rows_csv(path: str, rows: List[Dict], keys: List[str]):
    """
    Write rows into a csv, replacing older rows that have the same values in `keys`
    (so running evaluate.py twice does not create duplicates).
    """
    import pandas as pd
    ensure_dir(os.path.dirname(path) or '.')
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new_str = new.astype(str)
        sig_new = set(map(tuple, new_str[keys].values))
        keep = [tuple(r) not in sig_new for r in old[keys].values]
        old = old[keep]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(path, index=False)


def log_test_access(results_dir: str, script: str, what: str):
    """Append a line to results/test_set_access_log.txt every time the final test set DS2 is used."""
    import datetime
    ensure_dir(results_dir)
    with open(os.path.join(results_dir, 'test_set_access_log.txt'), 'a') as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | {script} | {what}\n")
