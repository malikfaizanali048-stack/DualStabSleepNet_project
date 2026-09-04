"""
Dataset / DataLoader for DSSNet training.

Handles:
  - Subject-wise splitting (train/val/test never share a subject -- the
    #1 leakage risk flagged in REPRODUCTION_SPEC.md).
  - Per-channel z-normalization using TRAIN-SPLIT-ONLY statistics (Fig 1a's
    "Normalization" step), frozen and reapplied identically to val/test.
  - Class-balanced sampling for N1 (paper's biggest reported per-stage
    gain, e.g. +12.5% on SHHS -- naive random sampling under-represents
    it every batch since it's 3-14% of epochs, see Table I).
  - CachedSpectrogramDataset: loads precomputed data-domain-stabilized +
    STFT spectrograms produced by precompute_stabilized.py, so Stage 1/2
    training never has to run the frozen 12-step diffusion sampler.

ASSUMPTIONS (paper's exact split protocol not given in provided pages):
  - `subject_wise_kfold` with n_folds=10 as a reasonable default matching
    common sleep-staging literature convention. Swap in the paper's actual
    declared folds if you find them, and update REPRODUCTION_SPEC.md.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_manifest(processed_dir: str):
    with open(os.path.join(processed_dir, "manifest.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def person_id_from_recording(recording_id: str) -> str:
    """
    SC4001E0 and SC4002E0 are the SAME real person's two different nights
    (subject number = chars [3:5], night number = char [5]). Splitting must
    happen at the PERSON level -- grouping by recording_id directly would
    let both nights of one person land in different splits, which is
    exactly the leakage this whole subject-wise design exists to prevent.
    """
    return recording_id[3:5]


def _group_recordings_by_person(recording_ids):
    groups = {}
    for rid in recording_ids:
        pid = person_id_from_recording(rid)
        groups.setdefault(pid, []).append(rid)
    return groups


def subject_wise_split(recording_ids, val_frac=0.15, test_frac=0.15, seed=42):
    groups = _group_recordings_by_person(recording_ids)
    person_ids = list(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(person_ids)

    n = len(person_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_people = person_ids[:n_test]
    val_people = person_ids[n_test:n_test + n_val]
    train_people = person_ids[n_test + n_val:]

    train_subj = [rid for pid in train_people for rid in groups[pid]]
    val_subj = [rid for pid in val_people for rid in groups[pid]]
    test_subj = [rid for pid in test_people for rid in groups[pid]]
    return train_subj, val_subj, test_subj


def subject_wise_kfold(recording_ids, n_folds=10, seed=42):
    """Yields (train_recordings, val_recordings, test_recordings) per fold,
    split at the PERSON level so both nights of one person always stay
    together in the same split."""
    groups = _group_recordings_by_person(recording_ids)
    person_ids = list(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(person_ids)
    folds = np.array_split(person_ids, n_folds)
    for i in range(n_folds):
        test_people = list(folds[i])
        remaining_people = [p for j, f in enumerate(folds) if j != i for p in f]
        rng2 = np.random.RandomState(seed + i)
        rng2.shuffle(remaining_people)
        n_val = max(1, int(len(remaining_people) * 0.15))
        val_people = remaining_people[:n_val]
        train_people = remaining_people[n_val:]

        train_r = [rid for pid in train_people for rid in groups[pid]]
        val_r = [rid for pid in val_people for rid in groups[pid]]
        test_r = [rid for pid in test_people for rid in groups[pid]]
        yield train_r, val_r, test_r


class SleepEpochDataset(Dataset):
    """
    Loads all epochs from the given subject .npz files into memory
    (~2.2GB for full SleepEDF-78; switch to lazy per-file loading if you
    hit memory limits, e.g. later on SHHS which is ~3x larger).
    """
    def __init__(self, processed_dir: str, subject_ids: list):
        self.processed_dir = processed_dir
        self.subject_ids = subject_ids
        self.mean = None
        self.std = None
        self._normalized = False

        xs, ys = [], []
        for sid in subject_ids:
            data = np.load(os.path.join(processed_dir, f"{sid}.npz"))
            xs.append(data["x"])
            ys.append(data["y"])
        self.x = np.concatenate(xs, axis=0).astype(np.float32)
        self.y = np.concatenate(ys, axis=0)   # (N,)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """
        mean/std: shape (C,), computed from TRAIN split only.

        Normalizes the ENTIRE array once, vectorized -- not per __getitem__
        call, which was the original CPU bottleneck.
        """
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.x = (self.x - self.mean[None, :, None]) / (self.std[None, :, None] + 1e-8)
        self._normalized = True

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        return torch.from_numpy(x), torch.tensor(int(self.y[idx]), dtype=torch.long)

    def class_counts(self):
        return np.bincount(self.y, minlength=5)


class CachedSpectrogramDataset(Dataset):
    """
    Loads precomputed (data-domain-stabilized + STFT) spectrograms produced
    by src/training/precompute_stabilized.py. Used by Stage 1/2 training so
    the frozen 12-step diffusion sampler never has to run during those
    training loops -- it already ran exactly once, offline, during
    precompute.

    Exposes the same `.y` / `class_counts()` interface as SleepEpochDataset
    so make_balanced_sampler() works unmodified.
    """
    def __init__(self, cache_path: str):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.spec = data["spec"]        # (N, C, H, W) precomputed spectrograms
        self.labels = data["labels"]    # (N,) long tensor
        self.y = self.labels.numpy()    # numpy view for make_balanced_sampler compatibility

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spec[idx], self.labels[idx]

    def class_counts(self):
        return np.bincount(self.y, minlength=5)


def compute_normalization_stats(train_ds: SleepEpochDataset):
    """Per-channel mean/std over ALL train-split epochs only.

    NOTE: must be called BEFORE set_normalization() on train_ds, since
    set_normalization now overwrites train_ds.x in place with the
    normalized version.
    """
    x = train_ds.x.astype(np.float64)
    mean = x.mean(axis=(0, 2))
    std = x.std(axis=(0, 2))
    return mean.astype(np.float32), std.astype(np.float32)


def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Inverse-class-frequency sampling weights to counter N1 under-representation.

    Works with both SleepEpochDataset and CachedSpectrogramDataset -- both
    expose .y (numpy labels) and .class_counts().
    """
    counts = dataset.class_counts()
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[dataset.y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )
