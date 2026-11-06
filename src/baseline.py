"""
baseline.py
===========
Simple (non deep-learning) baselines for the 4-class beat problem, using the 360 normalised
samples of a beat as features:

    * Logistic Regression  (multinomial, class_weight='balanced')
    * Random Forest        (200 trees, class_weight='balanced_subsample')

They are fitted on the TRAINING records and scored on the VALIDATION records only.
The best one (validation macro-F1) is saved as ../models/baseline_best.joblib; its result on the
final test set DS2 is produced later by evaluate.py --final (never here).

The very small CNN baseline ("tiny_cnn", about 3k parameters) is trained with train.py:
    python train.py --config ../configs/tiny_cnn.yaml

Usage:
    python baseline.py --config ../configs/edge_cnn.yaml
    python baseline.py --config ../configs/edge_cnn.yaml --max-train 20000      # faster
"""

import os
import sys
import time
import argparse

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (load_config, load_split, compute_metrics, seed_everything, ensure_dir,
                   metrics_rows, upsert_rows_csv, save_json, format_metrics_line)


def make_baselines(seed: int):
    return {
        'logreg': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                class_weight='balanced_subsample', random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser(description='Train simple baselines (validation only)')
    ap.add_argument('--config', default='../configs/edge_cnn.yaml')
    ap.add_argument('--max-train', type=int, default=None, help='use only N random training beats (speed)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    paths = {k: os.path.normpath(os.path.join(base, v)) for k, v in cfg['paths'].items()}
    seed = cfg.get('seed', 42)
    seed_everything(seed)
    ensure_dir(paths['models_dir']); ensure_dir(paths['results_dir'])

    Xtr, ytr = load_split(paths['processed_dir'], 'train')
    Xva, yva = load_split(paths['processed_dir'], 'val')          # DS2 is not loaded here
    if args.max_train and args.max_train < len(Xtr):
        idx = np.random.default_rng(seed).choice(len(Xtr), args.max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f'Train {Xtr.shape}, validation {Xva.shape}')

    results = {}
    csv_path = os.path.join(paths['results_dir'], 'baseline_validation.csv')
    for name, clf in make_baselines(seed).items():
        t0 = time.time()
        clf.fit(Xtr, ytr)
        m = compute_metrics(yva, clf.predict(Xva))
        results[name] = m
        print(f"{name:14s} {format_metrics_line(m)}  ({time.time() - t0:.0f}s)")
        joblib.dump(clf, os.path.join(paths['models_dir'], f'baseline_{name}.joblib'))
        upsert_rows_csv(csv_path, metrics_rows(f'baseline_{name}', 'val', 'clean', m),
                        ['model', 'split', 'condition', 'class'])

    best = max(results, key=lambda k: results[k]['macro_f1'])
    joblib.dump(joblib.load(os.path.join(paths['models_dir'], f'baseline_{best}.joblib')),
                os.path.join(paths['models_dir'], 'baseline_best.joblib'))
    save_json({'best': best, 'validation_macro_f1': results[best]['macro_f1'],
               'all': {k: v['macro_f1'] for k, v in results.items()}},
              os.path.join(paths['models_dir'], 'baseline_best.json'))
    print(f'Best baseline on validation: {best} (macro-F1 {results[best]["macro_f1"]:.4f})')


if __name__ == '__main__':
    main()
