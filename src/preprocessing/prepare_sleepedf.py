"""
Preprocess Sleep-EDF (20 or 78 subject variant) PSG recordings into
per-subject .npz epoch files.

Works for BOTH SleepEDF-20 and SleepEDF-78 from the *same* raw directory --
SleepEDF-20 is just subjects SC4000-SC4019 within the SleepEDF-78 expanded
dataset, so this script derives the 20-subject subset via a filename filter
rather than requiring a separate download.

Design choices made explicit (log these -- they are exactly the kind of
under-specified detail that causes reproduction gaps):
  - 30-second epochs, aligned to the hypnogram annotation onsets.
  - 5-class label scheme: W=0, N1=1, N2=2, N3=3 (N3+N4 merged), REM=4.
  - Epochs labeled "Sleep stage ?" or "Movement time" are DROPPED.
  - Only the in-bed period is kept: epochs are trimmed to
    [lights-off - pad_minutes, lights-on + pad_minutes] using the first/last
    non-Wake annotated epoch, padded by `pad_minutes` (paper doesn't specify
    this; 30 min is the common convention from DeepSleepNet-style pipelines
    and prevents huge Wake-class imbalance from long pre/post recording Wake).
  - Each subject's epochs are saved to ONE file: subject-wise split boundaries
    must always fall on file boundaries, never mid-file, to avoid leakage.

Usage:
    python prepare_sleepedf.py --config ../../configs/sleepedf78.yaml
    python prepare_sleepedf.py --config ../../configs/sleepedf20.yaml
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml

try:
    import mne
except ImportError:
    sys.exit(
        "mne is required for EDF reading. Install with:\n"
        "  pip install mne --break-system-packages"
    )

STAGE_MAP_DEFAULT = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
DROP_LABELS = {"Sleep stage ?", "Movement time"}
EPOCH_SEC = 30
PAD_MINUTES = 30


def load_config(path: str) -> dict:
    """Loads a YAML config, merging in a `defaults:` parent chain if present."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        parent_path = os.path.join(os.path.dirname(path), cfg.pop("defaults"))
        parent_cfg = load_config(parent_path)
        cfg = deep_merge(parent_cfg, cfg)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_subject_range(subject_filter: str):
    """Parses a 'lo-hi' subject-number range string, e.g. '0-19' -> (0, 19)."""
    lo_str, hi_str = subject_filter.split("-")
    return int(lo_str), int(hi_str)


def find_subject_pairs(raw_dir: str, subject_filter: str):
    """
    Finds matching PSG/Hypnogram EDF file pairs, e.g.:
      SC4001E0-PSG.edf  <->  SC4001EC-Hypnogram.edf
    Returns list of (subject_id, psg_path, hyp_path), sorted, one pair per
    subject (paper convention: use one recording per subject when duplicates
    exist across nights -- adjust here if you want multi-night per subject).
    """
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*PSG.edf"), recursive=True))
    pairs = []
    for psg_path in psg_files:
        base = os.path.basename(psg_path)
        # Filenames look like SC4ssNE0-PSG.edf: ss=2-digit subject, N=1-digit night.
        m = re.match(r"SC4(\d{2})(\d)E", base)
        if not m:
            continue
        subject_num = int(m.group(1))   # e.g. 0-19 for SleepEDF-20, 0-82ish for -78
        subject_full = base.split("-PSG")[0].split("-")[0]  # e.g. SC4001E0, kept as the unique recording id

        if subject_filter != "all":
            lo, hi = parse_subject_range(subject_filter)
            if not (lo <= subject_num <= hi):
                continue

        hyp_candidates = glob.glob(
            os.path.join(os.path.dirname(psg_path), f"SC4{subject_num:02d}*Hypnogram.edf")
        )
        if not hyp_candidates:
            print(f"[WARN] No hypnogram found for {psg_path}, skipping.")
            continue
        pairs.append((subject_full, psg_path, hyp_candidates[0]))
    return pairs


def extract_epochs(psg_path, hyp_path, channels, sampling_rate_hz, stage_map):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if int(round(raw.info["sfreq"])) != sampling_rate_hz:
        raw.resample(sampling_rate_hz)

    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)

    ch_names = [channels["eeg"], channels["eog"], channels["emg"]]
    missing = [c for c in ch_names if c not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in {psg_path}. "
            f"Available: {raw.ch_names}. Update configs/*.yaml channel names."
        )
    raw.pick(ch_names)

    events, event_id = mne.events_from_annotations(
        raw, chunk_duration=EPOCH_SEC, verbose=False
    )
    inv_event_id = {v: k for k, v in event_id.items()}

    sfreq = raw.info["sfreq"]
    samples_per_epoch = int(EPOCH_SEC * sfreq)
    data = raw.get_data()  # (n_channels, n_samples)

    # Trim to lights-off/on window with padding, per subject, to avoid
    # excessive long-Wake-run imbalance (see module docstring).
    non_wake_sample_idx = [
        ev[0] for ev in events if inv_event_id[ev[2]] not in DROP_LABELS
        and inv_event_id[ev[2]] != "Sleep stage W"
    ]
    if non_wake_sample_idx:
        pad_samples = int(PAD_MINUTES * 60 * sfreq)
        lo = max(0, min(non_wake_sample_idx) - pad_samples)
        hi = max(non_wake_sample_idx) + pad_samples

    epochs_x, epochs_y = [], []
    for onset_sample, _, code in events:
        label_str = inv_event_id[code]
        if label_str in DROP_LABELS or label_str not in stage_map:
            continue
        if non_wake_sample_idx and not (lo <= onset_sample <= hi):
            continue
        end_sample = onset_sample + samples_per_epoch
        if end_sample > data.shape[1]:
            continue
        seg = data[:, onset_sample:end_sample]  # (3, samples_per_epoch)
        epochs_x.append(seg.astype(np.float32))
        epochs_y.append(stage_map[label_str])

    if not epochs_x:
        return None, None
    return np.stack(epochs_x), np.array(epochs_y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process first N subjects")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    os.makedirs(ds["processed_dir"], exist_ok=True)

    stage_map = ds.get("stage_map", STAGE_MAP_DEFAULT)
    pairs = find_subject_pairs(ds["raw_dir"], ds["subject_filter"])
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Found {len(pairs)} subject recordings for dataset '{ds['name']}'.")
    manifest = []
    for subject_id, psg_path, hyp_path in pairs:
        out_path = os.path.join(ds["processed_dir"], f"{subject_id}.npz")
        if os.path.exists(out_path):
            print(f"[SKIP] {subject_id} already processed.")
            manifest.append(subject_id)
            continue
        try:
            x, y = extract_epochs(
                psg_path, hyp_path, ds["channels"], ds["sampling_rate_hz"], stage_map
            )
        except Exception as e:
            print(f"[ERROR] {subject_id}: {e}")
            continue
        if x is None:
            print(f"[WARN] {subject_id}: no valid epochs extracted.")
            continue
        np.savez_compressed(out_path, x=x, y=y, subject_id=subject_id)
        manifest.append(subject_id)
        counts = np.bincount(y, minlength=5)
        print(f"[OK] {subject_id}: {len(y)} epochs "
              f"(W={counts[0]} N1={counts[1]} N2={counts[2]} N3={counts[3]} REM={counts[4]})")

    manifest_path = os.path.join(ds["processed_dir"], "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest))
    print(f"\nDone. {len(manifest)} subjects written to {ds['processed_dir']}")
    print(f"Manifest (use this for subject-wise splits): {manifest_path}")


if __name__ == "__main__":
    main()