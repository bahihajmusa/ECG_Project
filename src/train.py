"""
train.py
========
Train the EdgeECGNet 1D CNN (deep learning) on the prepared MIT-BIH beats.

Stages
------
1. FP32 training  : Adam, class-weighted cross-entropy (or balanced sampler), augmentation on the
                    TRAINING beats only, early stopping on the VALIDATION macro-F1.
2. QAT (optional) : quantization-aware training. Starts from the best FP32 weights, folds BatchNorm,
                    fake-quantizes weights and activations to 8 bit and fine-tunes with a smaller
                    learning rate. The best QAT epoch is selected on the VALIDATION macro-F1 measured
                    with the INT8 emulation on.

Rules of the project (enforced here)
------------------------------------
* The final test set DS2 (X_test.npy) is NEVER loaded by this file. Model selection uses
  validation records only.
* Everything is controlled by a YAML file (see ../configs/edge_cnn.yaml); the seed is fixed.

Usage
-----
    python train.py --config ../configs/edge_cnn.yaml
    python train.py --config ../configs/edge_cnn.yaml --set training.epochs=5 training.lr=0.003
    python train.py --config ../configs/edge_cnn.yaml --sweep ../configs/sweep.yaml   # model selection (4.4)

Outputs
-------
    ../models/<name>_fp32.pt            best FP32 model
    ../models/<name>_qat.pt             best QAT (INT8-emulated) model
    ../results/training_log_<name>.csv  one row per epoch
    ../results/training_curves_<name>.png
    ../results/training_summary_<name>.json
    ../results/model_selection.csv      (only with --sweep)
    ../configs/best_config.yaml         (only with --sweep: the winning configuration)
"""

import os
import sys
import csv
import copy
import time
import argparse

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ECGBeatDataset, create_balanced_sampler, _worker_init_fn
from model import (build_model, save_checkpoint, count_parameters, check_max78002_constraints,
                   architecture_markdown)
from utils import (seed_everything, load_config, deep_update, get_device, ensure_dir, load_split,
                   compute_metrics, save_json, CLASS_NAMES, NUM_CLASSES)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def apply_overrides(cfg: dict, overrides):
    """--set a.b.c=value  (value parsed as YAML so numbers/bools work)."""
    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def class_weights_from_labels(y: np.ndarray, power: float = 1.0) -> torch.Tensor:
    """weight_c = (total / (4 * count_c)) ** power, computed from TRAINING labels only."""
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    w = np.zeros(NUM_CLASSES, dtype=np.float64)
    nz = counts > 0
    w[nz] = (len(y) / (NUM_CLASSES * counts[nz])) ** power
    return torch.tensor(w, dtype=torch.float32)


def make_scheduler(optimizer, kind: str, epochs: int):
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    if kind == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.3)
    return None


def evaluate_split(model, X, y, device, criterion=None):
    """Validation pass: loss (optional), accuracy, macro-F1, per-class F1."""
    model.eval()
    preds, loss_sum = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            xb = torch.from_numpy(X[i:i + 2048]).unsqueeze(1).to(device)
            logits = model(xb)
            if criterion is not None:
                loss_sum += float(criterion(logits, torch.from_numpy(y[i:i + 2048]).to(device))) * len(xb)
            preds.append(logits.argmax(1).cpu().numpy())
    m = compute_metrics(y, np.concatenate(preds))
    m['loss'] = loss_sum / len(X) if criterion is not None else float('nan')
    return m


# ----------------------------------------------------------------------------
# One training stage (FP32 or QAT)
# ----------------------------------------------------------------------------

def run_stage(stage, model, loaders, data, cfg_t, epochs, lr, device, criterion, log_rows,
              patience, min_epochs, log_fn):
    """
    Train `model` for up to `epochs` epochs, keep the best weights by validation macro-F1.
    Returns (best_state_dict, best_metrics, best_epoch).
    """
    train_loader = loaders['train']
    Xv, yv = data['val']
    opt_name = cfg_t.get('optimizer', 'adam').lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg_t.get('weight_decay', 0.0))
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg_t.get('weight_decay', 1e-2))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg_t.get('weight_decay', 0.0))
    scheduler = make_scheduler(optimizer, cfg_t.get('lr_schedule', 'cosine'), epochs)
    clip = cfg_t.get('grad_clip', 0.0)

    best_state, best_f1, best_loss, best_epoch, bad = None, -1.0, float('inf'), 0, 0
    if stage == 'qat':                                  # score before any QAT update
        m0 = evaluate_split(model, Xv, yv, device, criterion)
        best_state, best_f1, best_loss = copy.deepcopy(model.state_dict()), m0['macro_f1'], m0['loss']
        log_fn(f"[{stage}] epoch  0 | val_loss {m0['loss']:.4f} val_acc {m0['accuracy']:.4f} val_macroF1 {m0['macro_f1']:.4f}")
        log_rows.append(_row(stage, 0, lr, float('nan'), m0))

    for epoch in range(1, epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if clip and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            run_loss += loss.item() * len(xb)
            n += len(xb)
        cur_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        m = evaluate_split(model, Xv, yv, device, criterion)
        log_rows.append(_row(stage, epoch, cur_lr, run_loss / n, m))
        select_from = cfg_t.get('select_from_epoch', 1) if stage == 'fp32' else 1
        improved = epoch >= select_from and (m['macro_f1'] > best_f1 + 1e-4 or (abs(m['macro_f1'] - best_f1) <= 1e-4 and m['loss'] < best_loss))
        if improved:
            best_state, best_f1, best_loss, best_epoch, bad = copy.deepcopy(model.state_dict()), m['macro_f1'], m['loss'], epoch, 0
        else:
            bad += 1
        log_fn(f"[{stage}] epoch {epoch:2d}/{epochs} | lr {cur_lr:.5f} | train_loss {run_loss / n:.4f} | "
               f"val_loss {m['loss']:.4f} val_acc {m['accuracy']:.4f} val_macroF1 {m['macro_f1']:.4f} "
               f"| F1 {' '.join(f'{c}:{v:.2f}' for c, v in zip(CLASS_NAMES, m['f1']))} "
               f"| {time.time() - t0:.1f}s{' *' if improved else ''}")
        if epoch >= min_epochs and bad >= patience:
            log_fn(f"[{stage}] early stopping: no validation macro-F1 improvement for {patience} epochs")
            break
    return best_state, best_f1, best_epoch


def _row(stage, epoch, lr, train_loss, m):
    return {'stage': stage, 'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': m['loss'],
            'val_accuracy': m['accuracy'], 'val_macro_f1': m['macro_f1'],
            **{f'val_f1_{c}': v for c, v in zip(CLASS_NAMES, m['f1'])}}


# ----------------------------------------------------------------------------
# Full run
# ----------------------------------------------------------------------------

def run_training(cfg: dict, name: str, verbose: bool = True, save: bool = True) -> dict:
    """Train FP32 (+ optional QAT) as described by cfg. Returns a summary dict."""
    paths = cfg['paths']
    results_dir, models_dir = ensure_dir(paths['results_dir']), ensure_dir(paths['models_dir'])
    seed = cfg.get('seed', 42)
    seed_everything(seed)
    device = get_device(cfg.get('device', 'auto'))
    log = print if verbose else (lambda *_: None)
    t = cfg['training']

    # ---- data: TRAIN and VALIDATION only (DS2 is never loaded here) ----
    Xtr, ytr = load_split(paths['processed_dir'], 'train')
    Xva, yva = load_split(paths['processed_dir'], 'val')
    log(f"Run '{name}' on {device} | train {Xtr.shape} val {Xva.shape} | seed {seed}")

    imbalance = t.get('imbalance', 'class_weights')       # class_weights | balanced_sampler | none
    train_ds = ECGBeatDataset(Xtr, ytr, augment=t.get('augment', True))
    sampler = create_balanced_sampler(train_ds, seed) if imbalance == 'balanced_sampler' else None
    g = torch.Generator().manual_seed(seed)
    nw = t.get('num_workers', 0)
    train_loader = DataLoader(train_ds, batch_size=t.get('batch_size', 128), sampler=sampler,
                              shuffle=sampler is None, generator=g, num_workers=nw,
                              worker_init_fn=_worker_init_fn if nw > 0 else None)
    weights = None
    if imbalance == 'class_weights':
        weights = class_weights_from_labels(ytr, t.get('class_weight_power', 1.0))
        log(f"Class weights (train only): {dict(zip(CLASS_NAMES, [round(float(w), 3) for w in weights]))}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None,
                                    label_smoothing=t.get('label_smoothing', 0.0))

    # ---- model ----
    model = build_model(cfg['model']).to(device)
    chk = check_max78002_constraints(model)
    log(f"Model: {count_parameters(model):,} trainable / {count_parameters(model, True):,} deployed parameters | "
        f"MAX78002 pre-check: {'OK' if chk['ok'] else chk['problems']}")
    if verbose:
        log(architecture_markdown(model))

    log_rows, summary = [], {'name': name, 'seed': seed, 'device': str(device), 'imbalance': imbalance}
    loaders, data = {'train': train_loader}, {'val': (Xva, yva)}
    t_start = time.time()

    # ---- stage 1: FP32 ----
    es = t.get('early_stopping', {})
    best_state, best_f1, best_ep = run_stage(
        'fp32', model, loaders, data, t, t.get('epochs', 60), t.get('lr', 1e-3), device, criterion, log_rows,
        es.get('patience', 12), es.get('min_epochs', 10), log)
    model.load_state_dict(best_state)
    summary.update({'fp32_best_val_macro_f1': best_f1, 'fp32_best_epoch': best_ep,
                    'parameters_trainable': count_parameters(model),
                    'parameters_deployed': count_parameters(model, True)})
    log(f"FP32 best: epoch {best_ep}, val macro-F1 {best_f1:.4f}")
    if save:
        save_checkpoint(os.path.join(models_dir, f'{name}_fp32.pt'), model, cfg['model'],
                        {'config': cfg, 'val_macro_f1': best_f1, 'epoch': best_ep, 'stage': 'fp32'})

    # ---- stage 2: QAT ----
    q = cfg.get('qat', {})
    if q.get('enabled', True):
        model.set_quant('qat')
        qat_state, qat_f1, qat_ep = run_stage(
            'qat', model, loaders, data, t, q.get('epochs', 20), t.get('lr', 1e-3) * q.get('lr_factor', 0.1),
            device, criterion, log_rows, q.get('patience', 8), 1, log)
        model.load_state_dict(qat_state)
        summary.update({'qat_best_val_macro_f1': qat_f1, 'qat_best_epoch': qat_ep})
        log(f"QAT best: epoch {qat_ep}, val macro-F1 (INT8 emulation) {qat_f1:.4f}")
        if save:
            save_checkpoint(os.path.join(models_dir, f'{name}_qat.pt'), model, cfg['model'],
                            {'config': cfg, 'val_macro_f1': qat_f1, 'epoch': qat_ep, 'stage': 'qat'})

    summary['train_seconds'] = round(time.time() - t_start, 1)
    if save:
        _write_log(os.path.join(results_dir, f'training_log_{name}.csv'), log_rows)
        _plot_curves(log_rows, os.path.join(results_dir, f'training_curves_{name}.png'), name)
        save_json({**summary, 'config': cfg}, os.path.join(results_dir, f'training_summary_{name}.json'))
    return summary


def _write_log(path, rows):
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def _plot_curves(rows, path, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(rows))
    stages = [r['stage'] for r in rows]
    axes[0].plot(x, [r['train_loss'] for r in rows], label='train loss')
    axes[0].plot(x, [r['val_loss'] for r in rows], label='validation loss')
    axes[1].plot(x, [r['val_macro_f1'] for r in rows], label='validation macro-F1')
    axes[1].plot(x, [r['val_accuracy'] for r in rows], label='validation accuracy', alpha=0.6)
    if 'qat' in stages:
        for ax in axes:
            ax.axvline(stages.index('qat'), color='gray', ls='--', label='QAT starts')
    axes[0].set_title(f'{name}: loss'); axes[1].set_title(f'{name}: validation metrics')
    for ax in axes:
        ax.set_xlabel('epoch (FP32 then QAT)'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# Model selection (report 4.4)
# ----------------------------------------------------------------------------

def run_sweep(base_cfg: dict, sweep_path: str):
    """
    Train every experiment of the sweep file with short FP32 runs, select by VALIDATION macro-F1
    (ties within 0.005 -> fewer parameters), write results/model_selection.csv and
    configs/best_config.yaml.
    """
    sweep = load_config(sweep_path)
    rows = []
    for exp in sweep['experiments']:
        cfg = copy.deepcopy(base_cfg)
        deep_update(cfg, copy.deepcopy(exp.get('overrides', {})))
        deep_update(cfg, copy.deepcopy(sweep.get('common_overrides', {})))
        cfg['qat'] = {**cfg.get('qat', {}), 'enabled': False}       # selection uses FP32 runs
        print(f"\n=== Experiment: {exp['name']} | overrides: {exp.get('overrides')} ===")
        s = run_training(cfg, f"sweep_{exp['name']}", verbose=False, save=False)
        row = {'experiment': exp['name'], 'overrides': str(exp.get('overrides')),
               'val_macro_f1': s['fp32_best_val_macro_f1'], 'best_epoch': s['fp32_best_epoch'],
               'parameters_deployed': s['parameters_deployed'], 'seconds': s['train_seconds']}
        rows.append(row)
        print(f"-> val macro-F1 {row['val_macro_f1']:.4f}, deployed parameters {row['parameters_deployed']:,}")

    best_f1 = max(r['val_macro_f1'] for r in rows)
    near = [r for r in rows if r['val_macro_f1'] >= best_f1 - sweep.get('tie_margin', 0.005)]
    winner = min(near, key=lambda r: r['parameters_deployed'])
    for r in rows:
        r['selected'] = r is winner
    results_dir = ensure_dir(base_cfg['paths']['results_dir'])
    _write_log(os.path.join(results_dir, 'model_selection.csv'), rows)

    best_cfg = copy.deepcopy(base_cfg)
    deep_update(best_cfg, copy.deepcopy(next(e for e in sweep['experiments'] if e['name'] == winner['experiment']).get('overrides', {})))
    out_dir = os.path.dirname(os.path.abspath(sweep_path))
    out = os.path.join(out_dir, 'best_config.yaml')
    for k, v in best_cfg['paths'].items():          # store portable relative paths again
        best_cfg['paths'][k] = os.path.relpath(v, out_dir)
    with open(out, 'w') as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"\nSelected: {winner['experiment']} (val macro-F1 {winner['val_macro_f1']:.4f}, "
          f"{winner['parameters_deployed']:,} parameters). Config written to {out}")
    return winner


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Train the EdgeECGNet 1D CNN')
    ap.add_argument('--config', default='../configs/edge_cnn.yaml')
    ap.add_argument('--name', default=None, help='run name (default: config file name)')
    ap.add_argument('--set', nargs='*', default=[], metavar='key=value', help='override config values')
    ap.add_argument('--sweep', default=None, help='sweep yaml: run model selection instead of one training')
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    # make relative paths work from any folder: they are relative to the config file location
    base = os.path.dirname(os.path.abspath(args.config))
    for k, v in cfg['paths'].items():
        if not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))

    if args.sweep:
        run_sweep(cfg, args.sweep)
        return
    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    summary = run_training(cfg, name)
    print('\nSummary:', {k: v for k, v in summary.items() if k != 'config'})


if __name__ == '__main__':
    main()
