"""
dataset.py

Usage:
    from dataset import load_splits, create_dataloaders
    d = load_splits('../data/processed')
    loaders = create_dataloaders(d['X_train'], d['y_train'], d['X_val'], d['y_val'],
                                 d['X_test'], d['y_test'], batch_size=64)
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

NUM_CLASSES = 4
SAMPLING_FREQ = 360


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int = 42):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(worker_id: int):
    """
    Give every DataLoader worker its own numpy/python seed
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# AUGMENTATION TRANSFORMS (training data only)
# ============================================================================

class AmplitudeScaling:
    """Multiply the beat by a random gain (gain differences)."""

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, x):
        return x * np.random.uniform(*self.scale_range)


class TemporalShift:
    """
    Shift the beat by up to +-max_shift samples (R-peak detection jitter).
    Edges are filled with the border value (not 0) so no artificial step
    appears, whatever the normalisation is.
    """

    def __init__(self, max_shift=3):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return x
        out = np.roll(x, shift)
        if shift > 0:
            out[:shift] = x[0]
        else:
            out[shift:] = x[-1]
        return out


class BaselineWander:
    """
    Add a slow sinusoid (0.1-0.5 Hz) to simulate respiration / movement drift.
    amplitude is relative to the peak-to-peak range of the beat.
    """

    def __init__(self, max_amplitude=0.1, freq_range=(0.1, 0.5), fs=SAMPLING_FREQ):
        self.max_amplitude = max_amplitude
        self.freq_range = freq_range
        self.fs = fs

    def __call__(self, x):
        t = np.arange(len(x)) / self.fs
        f = np.random.uniform(*self.freq_range)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0, self.max_amplitude) * (np.ptp(x) + 1e-10)
        return x + amp * np.sin(2 * np.pi * f * t + phase)


class AddGaussianNoise:
    """Add Gaussian noise with std = noise_factor * std(beat) (sensor noise)."""

    def __init__(self, noise_factor=0.02):
        self.noise_factor = noise_factor

    def __call__(self, x):
        s = np.std(x)
        if s < 1e-10:
            s = 0.01
        return x + np.random.normal(0, self.noise_factor * s, size=x.shape)


class Compose:
    """Apply several transforms one after the other."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def default_train_augmentation():
    """Augmentation pipeline used for the training set (document it in 3.7)."""
    return Compose([
        AmplitudeScaling((0.9, 1.1)),
        TemporalShift(max_shift=3),
        BaselineWander(max_amplitude=0.1),
        AddGaussianNoise(0.02),
    ])


# ============================================================================
# DATASET
# ============================================================================

class ECGBeatDataset(Dataset):
    """
    Dataset of fixed-length ECG beats.

    Parameters:
        X (np.ndarray): (num_beats, segment_length) float beats, already normalised
        y (np.ndarray): (num_beats,) integer labels 0..3
        augment (bool): apply augmentation (ONLY for the training set)
        transform (callable): custom augmentation; used instead of the default
                              when augment=True

    __getitem__ returns (beat, label):
        beat  : float32 tensor (1, segment_length)  -> input of a 1D CNN
        label : int64 scalar tensor
    """

    def __init__(self, X, y, augment=False, transform=None):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (num_beats, length), got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        if len(X) == 0:
            raise ValueError("Dataset cannot be empty")
        if y.min() < 0 or y.max() >= NUM_CLASSES:
            raise ValueError(f"Labels must be in [0, {NUM_CLASSES - 1}]")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.transform = (transform if transform is not None else default_train_augmentation()) \
            if augment else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        beat = self.X[idx].copy()
        if self.augment and self.transform is not None:
            beat = self.transform(beat).astype(np.float32, copy=False)
        return torch.from_numpy(beat).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

    def get_class_distribution(self):
        """{class_index: count} for all 4 classes (0 if missing)."""
        counts = np.bincount(self.y, minlength=NUM_CLASSES)
        return {c: int(counts[c]) for c in range(NUM_CLASSES)}

    def get_class_weights(self):
        """Balanced weights total / (4 * count); 0 for a class that is absent."""
        counts = np.bincount(self.y, minlength=NUM_CLASSES).astype(np.float64)
        w = np.zeros(NUM_CLASSES, dtype=np.float32)
        nz = counts > 0
        w[nz] = len(self.y) / (NUM_CLASSES * counts[nz])
        return torch.from_numpy(w)


# ============================================================================
# DATALOADER UTILITIES
# ============================================================================

def create_balanced_sampler(dataset, seed=42):
    """
    WeightedRandomSampler: every class is drawn about equally often per epoch.
    (Vectorised; the old per-sample Python loop was slow for ~50k beats.)
    """
    class_w = dataset.get_class_weights().numpy()
    sample_w = class_w[dataset.y]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights=torch.from_numpy(sample_w).double(),
                                 num_samples=len(dataset), replacement=True, generator=g)


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size=64, use_balanced_sampler=True, augment_train=True,
                       num_workers=0, seed=42):
    """
    Build train / val / test loaders.
    Only the training set is augmented. Val and test are never augmented
    or re-sampled.
    """
    train_dataset = ECGBeatDataset(X_train, y_train, augment=augment_train)
    val_dataset = ECGBeatDataset(X_val, y_val, augment=False)
    test_dataset = ECGBeatDataset(X_test, y_test, augment=False)

    sampler = create_balanced_sampler(train_dataset, seed) if use_balanced_sampler else None
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
                  worker_init_fn=_worker_init_fn if num_workers > 0 else None)

    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=(sampler is None),
                              generator=g, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader,
            'train_dataset': train_dataset, 'val_dataset': val_dataset,
            'test_dataset': test_dataset}


def load_splits(processed_dir):
    """Load the .npy files written by prepare_data.py."""
    names = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    return {n: np.load(os.path.join(processed_dir, f"{n}.npy")) for n in names}


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == '__main__':
    seed_everything(42)
    X = np.random.randn(100, 360).astype(np.float32)
    y = np.random.randint(0, 4, size=100)

    ds = ECGBeatDataset(X, y, augment=False)
    beat, label = ds[0]
    assert beat.shape == (1, 360) and beat.dtype == torch.float32
    print("Plain dataset OK:", beat.shape, label)

    ds_aug = ECGBeatDataset(X, y, augment=True)
    assert not torch.equal(ds_aug[0][0], ds_aug[0][0]), "augmentation should change the beat"
    print("Augmented dataset OK")

    # a split that misses a class must not crash
    ds_missing = ECGBeatDataset(X[y != 3], y[y != 3])
    print("Class weights with missing class:", ds_missing.get_class_weights())

    loaders = create_dataloaders(X, y, X[:20], y[:20], X[:20], y[:20], batch_size=16)
    b, l = next(iter(loaders['train']))
    assert b.shape == (16, 1, 360) and l.shape == (16,)
    print("Dataloaders OK:", len(loaders['train']), len(loaders['val']), len(loaders['test']))
    print("All tests passed!")
