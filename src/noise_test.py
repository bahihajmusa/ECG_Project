"""
noise_test.py
=============
Noise-robustness experiment (report section 5.4).

The FINAL model is tested on the DS2 test records, clean and with real ECG noise added at
12 dB, 6 dB and 0 dB SNR (MIT-BIH Noise Stress Test Database: bw = baseline wander,
em = electrode motion artifact, ma = muscle artifact).

How the noisy data is made (same rules as prepare_data.py, no leakage):
  * noise is added to the RAW ECG record, then the normal band-pass filter, beat cutting and
    per-beat normalisation are applied (like a real device would do);
  * only the SECOND HALF of every noise recording is used (the first half is reserved for
    training augmentation), so training and test never share noise samples;
  * every noisy set contains exactly the same beats, in the same order, as the clean test set.

Optional ablation (--ablate-filter): the same noisy ECG is processed WITHOUT the band-pass filter,
to see whether the preprocessing helps at test time.

Outputs (../results):
    noise_results.csv            one row per model x noise type x SNR x filter setting
    noise_results_table.md       report table 5.4 (mean over the three noise types + per type)
    noise_robustness.png         macro-F1 and accuracy versus noise level
    noise_per_class_f1.png       per-class F1 versus noise level
    noise_filter_ablation.png    (with --ablate-filter)

Usage:
    python noise_test.py --config ../configs/edge_cnn.yaml --final \
        --raw-dir ../data/raw --noise-dir ../data/nstdb --ablate-filter
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_data as pdp
from evaluate import collect_models, predict_with
from utils import (load_config, load_split, compute_metrics, get_device, ensure_dir, log_test_access,
                   CLASS_NAMES)

SNR_DEFAULT = [12, 6, 0]
NOISE_NAMES = {'bw': 'baseline wander', 'em': 'electrode motion', 'ma': 'muscle artifact'}


def load_proc_kwargs(processed_dir: str) -> dict:
    """Preprocessing settings used for the clean data (from preprocessing_config.json)."""
    p = os.path.join(processed_dir, 'preprocessing_config.json')
    if not os.path.exists(p):
        return {'normalization': 'minmax', 'clip_sigma': None, 'zero_phase': True}
    with open(p) as f:
        c = json.load(f)
    return {'normalization': c.get('normalization_method', 'minmax'), 'clip_sigma': c.get('clip_sigma'),
            'zero_phase': c.get('filter', {}).get('zero_phase', True)}


class NoisyTestBuilder:
    """Builds (and caches) noisy versions of DS2."""

    def __init__(self, processed_dir, raw_dir, noise_dir, seed, regenerate=False):
        self.processed_dir, self.raw_dir, self.noise_dir = processed_dir, raw_dir, noise_dir
        self.seed, self.regenerate = seed, regenerate
        self.proc_kwargs = load_proc_kwargs(processed_dir)
        self._clean_records = None
        self._cache = {}

    def clean_records(self):
        if self._clean_records is None:
            if not os.path.isdir(self.raw_dir):
                sys.exit(f'Raw MIT-BIH data not found in {self.raw_dir} (needed to add noise to the raw ECG). '
                         'Download it with prepare_data.py or pass --raw-dir.')
            self._clean_records = pdp.process_all_records(self.raw_dir, pdp.DS2_RECORDS, **self.proc_kwargs)
        return self._clean_records

    def get(self, noise_type: str, snr_db: float, filter_on: bool = True) -> np.ndarray:
        key = (noise_type, snr_db, filter_on)
        if key in self._cache:
            return self._cache[key]
        saved = os.path.join(self.processed_dir, f'X_test_{noise_type}_{int(snr_db)}dB.npy')
        if filter_on and os.path.exists(saved) and not self.regenerate:
            X = np.load(saved).astype(np.float32)
        else:
            if pdp._missing_records(self.noise_dir, pdp.NSTDB_RECORDS):
                if not pdp.download_nstdb(self.noise_dir):
                    sys.exit(f'NSTDB noise records not found in {self.noise_dir}.')
            out = pdp.build_noisy_test_sets(self.raw_dir, self.noise_dir, self.clean_records(), [noise_type],
                                            [snr_db], self.seed, apply_filter=filter_on, **self.proc_kwargs)
            X = out[f'X_test_{noise_type}_{int(snr_db)}dB']
        self._cache[key] = X
        return X

    def clean_unfiltered(self) -> np.ndarray:
        """Clean DS2 beats built without the band-pass filter (for the ablation)."""
        parts = []
        for rid in pdp.DS2_RECORDS:
            rec = self.clean_records().get(rid)
            if rec is None or rec['num_beats'] == 0:
                continue
            res = pdp.process_single_record(rid, self.raw_dir, apply_filter=False,
                                            only_r_peaks=set(rec['r_peak_indices'].tolist()), **{
                                                k: v for k, v in self.proc_kwargs.items()})
            parts.append(res['beats'])
        return np.vstack(parts).astype(np.float32)


def run_condition(models, X, y, device, extra):
    rows = []
    for label, kind, obj, _ in models:
        pred = predict_with(kind, obj, X, device)
        m = compute_metrics(y, pred)
        rows.append({'model': label, **extra, 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                     **{f'f1_{c}': m['f1'][i] for i, c in enumerate(CLASS_NAMES)},
                     **{f'recall_{c}': m['recall'][i] for i, c in enumerate(CLASS_NAMES)}})
    return rows


def snr_label(s):
    return 'Clean' if s is None or (isinstance(s, float) and np.isnan(s)) else f'{int(s)} dB'


def make_plots(df, results_dir, main_model, ablation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snrs = ['clean', 12, 6, 0]
    xt = ['Clean', '12 dB', '6 dB', '0 dB']
    d = df[df['filter'] == 'on']

    def series(model, noise, col):
        sub = d[(d.model == model)]
        out = []
        for s in snrs:
            if s == 'clean':
                out.append(sub[sub.condition == 'clean'][col].mean())
            else:
                q = sub[(sub.snr_db == s)]
                q = q[q.noise_type == noise] if noise != 'mean' else q
                out.append(q[col].mean())
        return out

    models = [m for m in ['baseline', 'tiny_cnn_fp32', 'edge_fp32', 'edge_int8_qat'] if m in set(d.model)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in [(axes[0], 'macro_f1', 'Macro-F1'), (axes[1], 'accuracy', 'Accuracy')]:
        for noise, col_ in [('bw', 'tab:blue'), ('em', 'tab:orange'), ('ma', 'tab:green')]:
            if noise in set(d.noise_type):
                ax.plot(xt, series(main_model, noise, col), marker='o', color=col_, label=f'{main_model}: {NOISE_NAMES[noise]}')
        ax.plot(xt, series(main_model, 'mean', col), marker='s', color='black', lw=2.2, label=f'{main_model}: mean of noise types')
        for m in models:
            if m != main_model:
                ax.plot(xt, series(m, 'mean', col), ls='--', marker='.', label=f'{m}: mean')
        ax.set_title(f'{title} versus noise level (DS2)'); ax.set_xlabel('SNR'); ax.set_ylabel(title)
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(os.path.join(results_dir, 'noise_robustness.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in CLASS_NAMES:
        ax.plot(xt, series(main_model, 'mean', f'f1_{c}'), marker='o', label=f'class {c}')
    ax.set_title(f'{main_model}: per-class F1 versus noise (mean of noise types)'); ax.set_xlabel('SNR'); ax.set_ylabel('F1')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(results_dir, 'noise_per_class_f1.png'), dpi=150); plt.close(fig)

    if ablation:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for flt, style in [('on', '-'), ('off', '--')]:
            dd = df[(df['filter'] == flt) & (df.model == main_model)]
            vals = [dd[dd.condition == 'clean'].macro_f1.mean()] + [dd[dd.snr_db == s].macro_f1.mean() for s in [12, 6, 0]]
            ax.plot(xt, vals, style, marker='o', label=f'band-pass filter {flt}')
        ax.set_title(f'{main_model}: does the band-pass filter help? (mean macro-F1)'); ax.set_xlabel('SNR'); ax.set_ylabel('Macro-F1')
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); fig.savefig(os.path.join(results_dir, 'noise_filter_ablation.png'), dpi=150); plt.close(fig)


def write_table(df, path, main_model):
    d = df[df['filter'] == 'on']
    lines = ['# Noise robustness (DS2 test set)\n']
    for model in [m for m in ['baseline', 'tiny_cnn_fp32', 'edge_fp32', 'edge_int8_ptq', 'edge_int8_qat'] if m in set(d.model)]:
        sub = d[d.model == model]
        lines.append(f'## {model}: mean over the noise types (bw, em, ma)\n')
        lines.append('| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |')
        lines.append('|---|---|---|---|---|---|---|')
        for s in ['clean', 12, 6, 0]:
            q = sub[sub.condition == 'clean'] if s == 'clean' else sub[sub.snr_db == s]
            if len(q):
                lines.append(f"| {'Clean' if s == 'clean' else f'{s} dB'} | {q.accuracy.mean():.4f} | {q.macro_f1.mean():.4f} | "
                             + ' | '.join(f'{q[f"f1_{c}"].mean():.3f}' for c in CLASS_NAMES) + ' |')
        lines.append('')
    sub = d[d.model == main_model]
    lines.append(f'## {main_model}: per noise type\n')
    lines.append('| Noise type | SNR | Accuracy | Macro F1 |')
    lines.append('|---|---|---|---|')
    for noise in ['bw', 'em', 'ma']:
        for s in [12, 6, 0]:
            q = sub[(sub.noise_type == noise) & (sub.snr_db == s)]
            if len(q):
                lines.append(f'| {NOISE_NAMES[noise]} ({noise}) | {s} dB | {q.accuracy.iloc[0]:.4f} | {q.macro_f1.iloc[0]:.4f} |')
    if (df['filter'] == 'off').any():
        lines.append('\n## Band-pass filter ablation (main model, mean macro-F1)\n')
        lines.append('| Condition | filter ON | filter OFF |')
        lines.append('|---|---|---|')
        for s in ['clean', 12, 6, 0]:
            vals = []
            for flt in ['on', 'off']:
                dd = df[(df['filter'] == flt) & (df.model == main_model)]
                q = dd[dd.condition == 'clean'] if s == 'clean' else dd[dd.snr_db == s]
                vals.append(f'{q.macro_f1.mean():.4f}' if len(q) else '-')
            lines.append(f"| {'Clean' if s == 'clean' else f'{s} dB'} | {vals[0]} | {vals[1]} |")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description='Noise robustness test on DS2 (final evaluation)')
    ap.add_argument('--config', default='../configs/edge_cnn.yaml')
    ap.add_argument('--name', default='edge_cnn')
    ap.add_argument('--tiny-name', default='tiny_cnn')
    ap.add_argument('--final', action='store_true', help='required: this uses the final test set DS2')
    ap.add_argument('--raw-dir', default=None, help='raw MIT-BIH folder (default: ../data/raw next to the config)')
    ap.add_argument('--noise-dir', default=None, help='NSTDB folder (default: ../data/nstdb next to the config)')
    ap.add_argument('--noise-types', nargs='+', default=['bw', 'em', 'ma'])
    ap.add_argument('--snr-db', nargs='+', type=float, default=SNR_DEFAULT)
    ap.add_argument('--ablate-filter', action='store_true', help='also test without the band-pass filter')
    ap.add_argument('--regenerate', action='store_true', help='ignore saved X_test_<noise>_<snr>dB.npy files')
    ap.add_argument('--main-model', default='edge_int8_qat')
    ap.add_argument('--ai8x-preds', default=None)
    args = ap.parse_args()

    if not args.final:
        sys.exit('The noise test uses the final test set DS2. Add --final when all models are decided.')

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    paths = {k: os.path.normpath(os.path.join(base, v)) for k, v in cfg['paths'].items()}
    root = os.path.dirname(paths['processed_dir'])
    raw_dir = args.raw_dir or os.path.join(root, 'raw')
    noise_dir = args.noise_dir or os.path.join(root, 'nstdb')
    results_dir = ensure_dir(paths['results_dir'])
    device = get_device(cfg.get('device', 'auto'))

    _, y = load_split(paths['processed_dir'], 'test')
    Xclean, _ = load_split(paths['processed_dir'], 'test')
    models = collect_models(paths, args.name, args.tiny_name, device)
    models = [m for m in models if m[0] != 'edge_int8_ai8x']
    if not models:
        sys.exit('No trained models found.')
    main_model = args.main_model if args.main_model in [m[0] for m in models] else 'edge_fp32'
    builder = NoisyTestBuilder(paths['processed_dir'], raw_dir, noise_dir, cfg.get('seed', 42), args.regenerate)

    rows = []
    rows += run_condition(models, Xclean, y, device, {'noise_type': 'none', 'snr_db': np.nan, 'condition': 'clean', 'filter': 'on'})
    for noise in args.noise_types:
        for snr in args.snr_db:
            X = builder.get(noise, snr, True)
            assert len(X) == len(y), 'noisy set and clean labels have different length'
            rows += run_condition(models, X, y, device, {'noise_type': noise, 'snr_db': snr, 'condition': f'{noise}_{int(snr)}dB', 'filter': 'on'})
            r = [x for x in rows[-len(models):] if x['model'] == main_model][0]
            print(f"{noise} @ {int(snr):2d} dB | {main_model}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")
    if args.ablate_filter:
        Xu = builder.clean_unfiltered()
        rows += run_condition(models, Xu, y, device, {'noise_type': 'none', 'snr_db': np.nan, 'condition': 'clean', 'filter': 'off'})
        for noise in args.noise_types:
            for snr in args.snr_db:
                X = builder.get(noise, snr, False)
                rows += run_condition(models, X, y, device, {'noise_type': noise, 'snr_db': snr, 'condition': f'{noise}_{int(snr)}dB', 'filter': 'off'})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, 'noise_results.csv'), index=False)
    write_table(df, os.path.join(results_dir, 'noise_results_table.md'), main_model)
    make_plots(df, results_dir, main_model, args.ablate_filter)
    log_test_access(results_dir, 'noise_test.py', f'noise robustness ({", ".join(args.noise_types)} at {args.snr_db} dB)')

    d = df[(df['filter'] == 'on') & (df.model == main_model)]
    print('\nMean over noise types (main model: %s)' % main_model)
    for s in ['clean', 12, 6, 0]:
        q = d[d.condition == 'clean'] if s == 'clean' else d[d.snr_db == s]
        print(f"  {'Clean' if s == 'clean' else f'{s} dB':6s} accuracy {q.accuracy.mean():.4f}  macro-F1 {q.macro_f1.mean():.4f}")
    print(f'Results written to {results_dir}')


if __name__ == '__main__':
    main()
