"""
error_analysis.py
=================
Error analysis of the final model on the DS2 test set (report section 5.5).

It answers: which beats are misclassified, how, in which records, and what do they look like?
It also compares clean and strongly noisy conditions.

Outputs (../results):
    error_analysis.md              tables + automatically written facts (confusions, per-symbol and
                                   per-record error rates, noise effects) to help you write the discussion
    error_examples_<kind>.png      example misclassified beats (+ raw ECG context if raw data is available)
    error_mean_beats.png           average beat of each class: correct versus misclassified
    error_noise_examples.png       beats that are correct when clean but wrong with noise
    confusion_matrix_noise_<n>_<snr>dB.png

Usage:
    python error_analysis.py --config ../configs/edge_cnn.yaml --final
    python error_analysis.py --config ../configs/edge_cnn.yaml --final --noise ma --snr 0
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_data as pdp
from evaluate import collect_models
from noise_test import NoisyTestBuilder
from utils import (load_config, load_split, load_meta, compute_metrics, get_device, ensure_dir,
                   log_test_access, plot_confusion_matrix, CLASS_NAMES, NUM_CLASSES, predict)

COLORS = {'N': '#2ecc71', 'S': '#3498db', 'V': '#e74c3c', 'F': '#f39c12'}


def predict_probs(kind, obj, X, device):
    if kind == 'torch':
        return predict(obj, X, device, return_probs=True)
    if kind == 'sklearn':
        p = obj.predict_proba(X)
        return p.argmax(1), p
    sys.exit('error_analysis needs a model object (not saved predictions).')


class RawContext:
    """Reads the filtered raw ECG around a beat (for the example plots). Optional."""

    def __init__(self, raw_dir):
        self.raw_dir, self.cache = raw_dir, {}
        self.ok = raw_dir is not None and os.path.isdir(raw_dir)

    def get(self, rid, r_peak, half=540):
        if not self.ok:
            return None
        try:
            if rid not in self.cache:
                rec = pdp.wfdb.rdrecord(os.path.join(self.raw_dir, rid))
                sig, _, _ = pdp.select_ecg_channel(rec)
                self.cache[rid] = pdp.apply_bandpass_filter(np.nan_to_num(sig), fs=rec.fs)
            s = self.cache[rid]
            a, b = max(0, r_peak - half), min(len(s), r_peak + half)
            return np.arange(a, b) - r_peak, s[a:b]
        except Exception:
            return None


def plot_error_examples(idx, X, y, pred, probs, meta, ctx, title, path, n=6):
    """Grid: top row = model input beat; bottom row = raw filtered ECG context (if available)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idx = idx[:n]
    if len(idx) == 0:
        return False
    rows = 2 if ctx.ok else 1
    fig, axes = plt.subplots(rows, len(idx), figsize=(3.1 * len(idx), 2.6 * rows), squeeze=False)
    t = np.arange(-180, 180)
    for j, i in enumerate(idx):
        t_c, p_c = CLASS_NAMES[y[i]], CLASS_NAMES[pred[i]]
        ax = axes[0, j]
        ax.plot(t, X[i], color=COLORS[t_c]); ax.axvline(0, color='k', ls=':', alpha=0.4)
        rid = meta['record_ids'][i] if meta['record_ids'] is not None else '?'
        sym = meta['symbols'][i] if meta['symbols'] is not None else '?'
        ax.set_title(f'true {t_c} ({sym}) -> pred {p_c}\nrecord {rid}, p={probs[i][pred[i]]:.2f}', fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        if rows == 2:
            c = ctx.get(str(rid), int(meta['r_peaks'][i])) if meta['r_peaks'] is not None else None
            ax2 = axes[1, j]
            if c is not None:
                ax2.plot(c[0] / 360.0, c[1], color='gray', lw=0.8)
                ax2.axvspan(-0.5, 0.5, color=COLORS[t_c], alpha=0.15)
                ax2.set_xlabel('s around R', fontsize=7)
            else:
                ax2.axis('off')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def pick(idx_mask, probs, pred, n, rng):
    """Half of the examples = most confident errors, half = random errors."""
    idx = np.where(idx_mask)[0]
    if len(idx) == 0:
        return idx
    conf = probs[idx, pred[idx]]
    order = idx[np.argsort(-conf)]
    k = min(n // 2 + n % 2, len(order))
    chosen = list(order[:k])
    rest = [i for i in idx if i not in set(chosen)]
    if rest:
        chosen += list(rng.choice(rest, size=min(n - k, len(rest)), replace=False))
    return np.array(chosen, dtype=int)


def main():
    ap = argparse.ArgumentParser(description='Error analysis on the final test set')
    ap.add_argument('--config', default='../configs/edge_cnn.yaml')
    ap.add_argument('--name', default='edge_cnn')
    ap.add_argument('--tiny-name', default='tiny_cnn')
    ap.add_argument('--model', default='edge_int8_qat', help='which model to analyse')
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--noise-dir', default=None)
    ap.add_argument('--noise', default='ma', choices=['bw', 'em', 'ma'])
    ap.add_argument('--snr', type=float, default=0.0)
    ap.add_argument('--n-examples', type=int, default=6)
    args = ap.parse_args()
    if not args.final:
        sys.exit('Error analysis uses the final test set DS2: add --final (after all decisions are made).')

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    paths = {k: os.path.normpath(os.path.join(base, v)) for k, v in cfg['paths'].items()}
    root = os.path.dirname(paths['processed_dir'])
    raw_dir = args.raw_dir or os.path.join(root, 'raw')
    noise_dir = args.noise_dir or os.path.join(root, 'nstdb')
    results_dir = ensure_dir(paths['results_dir'])
    device = get_device(cfg.get('device', 'auto'))
    rng = np.random.default_rng(cfg.get('seed', 42))

    X, y = load_split(paths['processed_dir'], 'test')
    meta = load_meta(paths['processed_dir'], 'test')
    models = {m[0]: m for m in collect_models(paths, args.name, args.tiny_name, device)}
    label = args.model if args.model in models else ('edge_fp32' if 'edge_fp32' in models else None)
    if label is None:
        sys.exit('No suitable model found. Train first.')
    _, kind, obj, _ = models[label]
    pred, probs = predict_probs(kind, obj, X, device)
    m = compute_metrics(y, pred)
    cm = np.array(m['confusion'])
    ctx = RawContext(raw_dir)
    md = [f'# Error analysis ({label}, DS2 test set)\n',
          f"Beats: {len(y)} | accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | errors: {int((pred != y).sum())}\n"]

    # ---- 1. confusion structure ----
    md.append('## 1. Most frequent confusions\n')
    md.append('| True | Predicted | Beats | % of the true class |')
    md.append('|---|---|---|---|')
    pairs = [(i, j, cm[i, j]) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j] > 0]
    for i, j, c in sorted(pairs, key=lambda p: -p[2])[:8]:
        md.append(f'| {CLASS_NAMES[i]} | {CLASS_NAMES[j]} | {c} | {100 * c / max(cm[i].sum(), 1):.1f}% |')
    md.append('')
    for i in range(1, NUM_CLASSES):
        if cm[i].sum() > 0:
            wrong = cm[i].sum() - cm[i, i]
            if wrong > 0:
                j = int(np.argmax(np.where(np.arange(NUM_CLASSES) == i, -1, cm[i])))
                md.append(f'* Class {CLASS_NAMES[i]}: {100 * cm[i, i] / cm[i].sum():.1f}% recall; '
                          f'{100 * cm[i, j] / wrong:.0f}% of its errors are predicted as {CLASS_NAMES[j]}.')
    md.append('')

    # ---- 2. per original symbol ----
    if meta['symbols'] is not None:
        md.append('## 2. Error rate per original MIT-BIH symbol\n')
        md.append('| Symbol | Class | Beats | Errors | Error rate | Most common wrong class |')
        md.append('|---|---|---|---|---|---|')
        syms = meta['symbols']
        for s in sorted(set(syms.tolist()), key=lambda s: -(syms == s).sum()):
            mask = syms == s
            wrong = mask & (pred != y)
            wc = Counter(pred[wrong].tolist()).most_common(1)
            md.append(f'| {s} | {CLASS_NAMES[int(y[mask][0])]} | {int(mask.sum())} | {int(wrong.sum())} | '
                      f'{100 * wrong.sum() / mask.sum():.1f}% | {CLASS_NAMES[wc[0][0]] if wc else "-"} |')
        md.append('')

    # ---- 3. per record ----
    if meta['record_ids'] is not None:
        md.append('## 3. Records with the highest error rate\n')
        md.append('| Record | Beats | Errors | Error rate | Class counts N/S/V/F |')
        md.append('|---|---|---|---|---|')
        rids = meta['record_ids']
        rows = []
        for r in sorted(set(rids.tolist())):
            mask = rids == r
            e = int((pred[mask] != y[mask]).sum())
            rows.append((r, int(mask.sum()), e, e / mask.sum(), np.bincount(y[mask], minlength=4)))
        for r, n, e, rate, bc in sorted(rows, key=lambda t: -t[2])[:8]:
            md.append(f'| {r} | {n} | {e} | {100 * rate:.1f}% | {"/".join(map(str, bc))} |')
        tot_err = sum(r[2] for r in rows)
        top3 = sum(sorted([r[2] for r in rows], reverse=True)[:3])
        md.append(f'\nThe 3 records with most errors hold {100 * top3 / max(tot_err, 1):.0f}% of all errors.\n')

    # ---- 4. example plots ----
    md.append('## 4. Example figures\n')
    kinds = [('N_to_S', (y == 0) & (pred == 1), 'True N predicted as S (false alarms for S)'),
             ('S_to_N', (y == 1) & (pred == 0), 'True S predicted as N (missed S)'),
             ('V_missed', (y == 2) & (pred != 2), 'True V that the model missed'),
             ('F_missed', (y == 3) & (pred != 3), 'True F that the model missed')]
    for name, mask, title in kinds:
        idx = pick(mask, probs, pred, args.n_examples, rng)
        out = os.path.join(results_dir, f'error_examples_{name}.png')
        if plot_error_examples(idx, X, y, pred, probs, meta, ctx, f'{label}: {title} ({int(mask.sum())} beats)', out, args.n_examples):
            md.append(f'* `error_examples_{name}.png`: {title} ({int(mask.sum())} beats in total)')
    md.append('')

    # average beat: correct vs wrong
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    t = np.arange(-180, 180)
    for c in range(NUM_CLASSES):
        ok, bad = (y == c) & (pred == c), (y == c) & (pred != c)
        if ok.any():
            axes[c].plot(t, X[ok].mean(0), color=COLORS[CLASS_NAMES[c]], label=f'correct (n={int(ok.sum())})')
        if bad.any():
            axes[c].plot(t, X[bad].mean(0), color='k', ls='--', label=f'misclassified (n={int(bad.sum())})')
        axes[c].set_title(f'class {CLASS_NAMES[c]}: average beat'); axes[c].legend(fontsize=7); axes[c].grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(results_dir, 'error_mean_beats.png'), dpi=140); plt.close(fig)
    md.append('* `error_mean_beats.png`: average beat of each class, correctly classified versus misclassified\n')

    # ---- 5. noise ----
    md.append(f'## 5. Effect of noise ({args.noise}, {int(args.snr)} dB SNR)\n')
    try:
        builder = NoisyTestBuilder(paths['processed_dir'], raw_dir, noise_dir, cfg.get('seed', 42))
        Xn = builder.get(args.noise, args.snr, True)
        pred_n, probs_n = predict_probs(kind, obj, Xn, device)
        mn = compute_metrics(y, pred_n)
        plot_confusion_matrix(mn['confusion'], os.path.join(results_dir, f'confusion_matrix_noise_{args.noise}_{int(args.snr)}dB.png'),
                              f'{label}, {args.noise} {int(args.snr)} dB')
        md.append(f"Clean: accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}. "
                  f"Noisy: accuracy {mn['accuracy']:.4f}, macro-F1 {mn['macro_f1']:.4f}.\n")
        flipped = (pred == y) & (pred_n != y)
        fixed = (pred != y) & (pred_n == y)
        md.append(f'* {int(flipped.sum())} beats were correct when clean and wrong with noise; {int(fixed.sum())} went the other way.')
        wrong_to = Counter(pred_n[flipped].tolist())
        if wrong_to:
            md.append('* Where the noise pushes the predictions: ' + ', '.join(
                f'{CLASS_NAMES[k]} ({100 * v / flipped.sum():.0f}%)' for k, v in wrong_to.most_common()))
        md.append('\n| Class | Recall clean | Recall noisy | F1 clean | F1 noisy |')
        md.append('|---|---|---|---|---|')
        for c in range(NUM_CLASSES):
            md.append(f"| {CLASS_NAMES[c]} | {m['recall'][c]:.3f} | {mn['recall'][c]:.3f} | {m['f1'][c]:.3f} | {mn['f1'][c]:.3f} |")
        idx = pick(flipped, probs_n, pred_n, args.n_examples, rng)
        if len(idx):
            fig, axes = plt.subplots(1, len(idx), figsize=(3.1 * len(idx), 2.7), squeeze=False)
            for j, i in enumerate(idx):
                axes[0, j].plot(t, X[i], color='gray', label='clean')
                axes[0, j].plot(t, Xn[i], color=COLORS[CLASS_NAMES[y[i]]], lw=0.9, label='noisy')
                axes[0, j].set_title(f'true {CLASS_NAMES[y[i]]}: clean ok -> noisy {CLASS_NAMES[pred_n[i]]}', fontsize=8)
                axes[0, j].set_ylim(-1.3, 1.3)
            axes[0, 0].legend(fontsize=7)
            plt.tight_layout(); fig.savefig(os.path.join(results_dir, 'error_noise_examples.png'), dpi=140); plt.close(fig)
            md.append('\n* `error_noise_examples.png`: beats that are correct when clean but wrong with noise')
    except SystemExit as e:
        md.append(f'(noise part skipped: {e})')

    md.append('\n## 6. Points to discuss in the report\n')
    md.append('* Which class is hardest and why (small number of training beats, similar shape to N, patient differences)?')
    md.append('* Are the errors concentrated in a few records (section 3)?')
    md.append('* Do errors look like real ambiguity, label noise, or signal artifacts (see the example figures)?')
    md.append('* What does noise change: which classes lose most, and where do the wrong predictions go?')
    with open(os.path.join(results_dir, 'error_analysis.md'), 'w') as f:
        f.write('\n'.join(md))
    log_test_access(results_dir, 'error_analysis.py', f'error analysis of {label}')
    print('\n'.join(md[:30]))
    print(f'\nWritten: {os.path.join(results_dir, "error_analysis.md")} and figures')


if __name__ == '__main__':
    main()
