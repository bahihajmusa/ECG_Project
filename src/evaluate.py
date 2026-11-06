"""
evaluate.py
===========
Evaluate the trained models and write the tables of report section 5.

Models that are evaluated (if their files exist):
    baseline        ../models/baseline_best.joblib      Logistic Regression or Random Forest
    tiny_cnn_fp32   ../models/tiny_cnn_fp32.pt          very small CNN (deep-learning baseline)
    edge_fp32       ../models/<name>_fp32.pt            proposed EdgeECGNet, FP32
    edge_int8_ptq   same FP32 weights, post-training INT8 emulation (no retraining)
    edge_int8_qat   ../models/<name>_qat.pt             quantization-aware trained, INT8 emulation
    edge_int8_ai8x  predictions of the official ai8x quantized model (optional, --ai8x-preds)

Rules of the project:
  * --split val  : validation records (use this while developing).
  * --split test : the FINAL test set DS2. You must add --final, and every use is written to
                   results/test_set_access_log.txt. Never use test results to choose anything.

Outputs (in ../results):
    metrics.csv                         accuracy, macro-F1, per-class precision/recall/F1 for every model
    confusion_matrix_<model>_<split>.png/.csv   (+ confusion_matrix.png/.csv = main model)
    tables_<split>.md                   report tables 5.1, 5.2 and 5.3 in markdown
    preds/<model>_<split>.npy           predictions (used by error_analysis.py)

Usage:
    python evaluate.py --config ../configs/edge_cnn.yaml --split val
    python evaluate.py --config ../configs/edge_cnn.yaml --split test --final
"""

import os
import sys
import argparse

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_model, count_parameters, check_max78002_constraints
from utils import (load_config, load_split, compute_metrics, predict, get_device, ensure_dir,
                   metrics_rows, upsert_rows_csv, plot_confusion_matrix, save_confusion_csv,
                   log_test_access, format_metrics_line, save_json, CLASS_NAMES)


def sklearn_param_count(clf) -> int:
    if hasattr(clf, 'coef_'):
        return int(clf.coef_.size + clf.intercept_.size)
    if hasattr(clf, 'estimators_'):
        return int(sum(e.tree_.node_count for e in clf.estimators_))
    return -1


def collect_models(paths, name, tiny_name, device, ai8x_preds=None):
    """Return a list of (label, kind, object, parameter_count)."""
    mdir = paths['models_dir']
    out = []
    p = os.path.join(mdir, 'baseline_best.joblib')
    if os.path.exists(p):
        clf = joblib.load(p)
        out.append(('baseline', 'sklearn', clf, sklearn_param_count(clf)))
    p = os.path.join(mdir, f'{tiny_name}_fp32.pt')
    if os.path.exists(p):
        m = load_model(p, device)
        out.append(('tiny_cnn_fp32', 'torch', m, count_parameters(m, True)))
    p = os.path.join(mdir, f'{name}_fp32.pt')
    if os.path.exists(p):
        m = load_model(p, device)
        out.append(('edge_fp32', 'torch', m, count_parameters(m, True)))
        mp = load_model(p, device, quant='ptq')
        out.append(('edge_int8_ptq', 'torch', mp, count_parameters(mp, True)))
    p = os.path.join(mdir, f'{name}_qat.pt')
    if os.path.exists(p):
        m = load_model(p, device)
        out.append(('edge_int8_qat', 'torch', m, count_parameters(m, True)))
    if ai8x_preds is not None:
        out.append(('edge_int8_ai8x', 'preds', np.load(ai8x_preds), count_parameters(out[-1][2], True) if out else -1))
    return out


def predict_with(kind, obj, X, device):
    if kind == 'sklearn':
        return obj.predict(X)
    if kind == 'torch':
        return predict(obj, X, device)
    return obj          # precomputed predictions


def fmt(v, nd=4):
    return f'{v:.{nd}f}'


def write_tables(path, split, results, params, main_model):
    """Markdown tables 5.1 / 5.2 / 5.3."""
    lines = [f'# Evaluation tables ({split} set)\n']
    cols = [c for c in ['baseline', 'tiny_cnn_fp32', 'edge_fp32', 'edge_int8_ptq', 'edge_int8_qat', 'edge_int8_ai8x'] if c in results]
    names = {'baseline': 'Baseline (best classical)', 'tiny_cnn_fp32': 'Tiny CNN (FP32)', 'edge_fp32': 'Edge model FP32',
             'edge_int8_ptq': 'INT8 post-training', 'edge_int8_qat': 'INT8 QAT (emulated)', 'edge_int8_ai8x': 'INT8 ai8x (official)'}
    lines.append('## 5.1 Overall results\n')
    lines.append('| Metric | ' + ' | '.join(names[c] for c in cols) + ' |')
    lines.append('|---|' + '---|' * len(cols))
    lines.append('| Accuracy | ' + ' | '.join(fmt(results[c]['accuracy']) for c in cols) + ' |')
    lines.append('| Macro F1 | ' + ' | '.join(fmt(results[c]['macro_f1']) for c in cols) + ' |')
    lines.append('| Model parameters | ' + ' | '.join(f'{params[c]:,}' if params[c] >= 0 else 'n/a' for c in cols) + ' |')
    lines.append('\nBaseline parameter count: Logistic Regression = weights + biases; Random Forest = total tree nodes.\n')

    lines.append('## 5.2 Per-class performance\n')
    for c in cols:
        m = results[c]
        lines.append(f'### {names[c]}\n')
        lines.append('| Class | Precision | Sensitivity/Recall | F1 | Support |')
        lines.append('|---|---|---|---|---|')
        for i, cl in enumerate(CLASS_NAMES):
            lines.append(f"| {cl} | {fmt(m['precision'][i])} | {fmt(m['recall'][i])} | {fmt(m['f1'][i])} | {m['support'][i]} |")
        lines.append('')

    if 'edge_fp32' in results:
        lines.append('## 5.3 Quantization analysis\n')
        f32 = results['edge_fp32']['macro_f1']
        for key, label in [('edge_int8_ptq', 'INT8 post-training (PTQ)'), ('edge_int8_qat', 'INT8 QAT (emulated)'),
                           ('edge_int8_ai8x', 'INT8 ai8x (official toolchain)')]:
            if key in results:
                lines.append(f'* FP32 macro F1: {fmt(f32)} | {label} macro F1: {fmt(results[key]["macro_f1"])} | '
                             f'difference: {results[key]["macro_f1"] - f32:+.4f}')
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description='Evaluate models on validation or the final test set')
    ap.add_argument('--config', default='../configs/edge_cnn.yaml')
    ap.add_argument('--name', default='edge_cnn', help='run name used by train.py for the edge model')
    ap.add_argument('--tiny-name', default='tiny_cnn')
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--final', action='store_true', help='required to touch the final test set DS2')
    ap.add_argument('--ai8x-preds', default=None, help='npy file with predictions of the official ai8x INT8 model')
    ap.add_argument('--main-model', default='edge_int8_qat', help='model whose confusion matrix is saved as confusion_matrix.png')
    args = ap.parse_args()

    if args.split == 'test' and not args.final:
        sys.exit('Refusing to evaluate on the final test set DS2 without --final.\n'
                 'The test set must only be used once, after all models and settings are decided.')

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    paths = {k: os.path.normpath(os.path.join(base, v)) for k, v in cfg['paths'].items()}
    results_dir = ensure_dir(paths['results_dir'])
    device = get_device(cfg.get('device', 'auto'))

    X, y = load_split(paths['processed_dir'], args.split)
    print(f'Split: {args.split} | beats: {len(y)} | device: {device}')
    models = collect_models(paths, args.name, args.tiny_name, device, args.ai8x_preds)
    if not models:
        sys.exit('No trained models found in ' + paths['models_dir'] + ' (run baseline.py and train.py first).')

    results, params = {}, {}
    ensure_dir(os.path.join(results_dir, 'preds'))
    for label, kind, obj, n_params in models:
        pred = predict_with(kind, obj, X, device)
        m = compute_metrics(y, pred)
        results[label], params[label] = m, n_params
        print(f'{label:16s} {format_metrics_line(m)}')
        np.save(os.path.join(results_dir, 'preds', f'{label}_{args.split}.npy'), pred)
        upsert_rows_csv(os.path.join(results_dir, 'metrics.csv'),
                        metrics_rows(label, args.split, 'clean', m, n_params),
                        ['model', 'split', 'condition', 'class'])
        plot_confusion_matrix(m['confusion'], os.path.join(results_dir, f'confusion_matrix_{label}_{args.split}.png'),
                              f'{label} ({args.split})')
        save_confusion_csv(m['confusion'], os.path.join(results_dir, f'confusion_matrix_{label}_{args.split}.csv'))

    # main confusion matrix (name required by the submission structure)
    main_label = args.main_model if args.main_model in results else ('edge_fp32' if 'edge_fp32' in results else list(results)[-1])
    if args.split == 'test':
        plot_confusion_matrix(results[main_label]['confusion'], os.path.join(results_dir, 'confusion_matrix.png'),
                              f'{main_label} (test, DS2)')
        save_confusion_csv(results[main_label]['confusion'], os.path.join(results_dir, 'confusion_matrix.csv'))

    # agreement between FP32 and INT8 predictions
    if 'edge_fp32' in results:
        p32 = np.load(os.path.join(results_dir, 'preds', f'edge_fp32_{args.split}.npy'))
        for key in ('edge_int8_ptq', 'edge_int8_qat', 'edge_int8_ai8x'):
            if key in results:
                pq = np.load(os.path.join(results_dir, 'preds', f'{key}_{args.split}.npy'))
                print(f'Agreement FP32 vs {key}: {100 * np.mean(p32 == pq):.2f}% of beats get the same class')

    write_tables(os.path.join(results_dir, f'tables_{args.split}.md'), args.split, results, params, main_label)
    save_json(results, os.path.join(results_dir, f'evaluation_{args.split}.json'))

    if 'edge_fp32' in results:
        m = [mod for lab, k, mod, _ in models if lab == 'edge_fp32'][0]
        print('MAX78002 pre-check:', check_max78002_constraints(m))
    if args.split == 'test':
        log_test_access(results_dir, 'evaluate.py', 'clean evaluation of ' + ', '.join(results))
    print(f'Tables written to {os.path.join(results_dir, f"tables_{args.split}.md")}')


if __name__ == '__main__':
    main()
