"""
Preprocess ISRUC-S3 (10 healthy subjects, single session each) into
per-subject .npz epoch files, matching the same output format as
prepare_sleepedf.py so downstream training code is dataset-agnostic.

Format (confirmed against real files):
  {raw_dir}/{subject_num}/{subject_num}.rec        <- PSG signal, EDF format
                                                        despite .rec extension
  {raw_dir}/{subject_num}/{subject_num}_1.txt      <- hypnogram, scorer 1
  {raw_dir}/{subject_num}/{subject_num}_2.txt      <- hypnogram, scorer 2
One hypnogram line = one 30s epoch label, 1:1 aligned with signal epochs.

No person-level grouping needed (unlike SleepEDF) -- one recording per subject.

Usage:
    python prepare_isruc.py --config ../../configs/isruc_s3.yaml
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_utils import load_config

try:
    import mne
except ImportError:
    sys.exit("mne is required. Install with: pip install mne --break-system-packages")

EPOCH_SEC = 30
PAD_MINUTES = 30  # lights-off trimming, same convention as prepare_sleepedf.py
PAD_EPOCHS = int(PAD_MINUTES * 60 / EPOCH_SEC)


def find_subject_dirs(raw_dir: str):
    subjects = []
    for name in sorted(os.listdir(raw_dir), key=lambda x: (len(x), x)):
        subject_dir = os.path.join(raw_dir, name)
        if not os.path.isdir(subject_dir):
            continue
        if not name.isdigit():
            continue
        rec_path = os.path.join(subject_dir, f"{name}.rec")
        if os.path.exists(rec_path):
            subjects.append(name)
    return subjects


def read_hypnogram(subject_dir: str, subject_num: str, scorer: int, stage_map: dict):
    hyp_path = os.path.join(subject_dir, f"{subject_num}_{scorer}.txt")
    with open(hyp_path) as f:
        raw_labels = [int(line.strip()) for line in f if line.strip()]

    unmapped = set(raw_labels) - set(stage_map.keys())
    if unmapped:
        raise ValueError(
            f"Unexpected label value(s) {unmapped} in {hyp_path} not in stage_map "
            f"{stage_map}. Do not silently proceed -- verify the label encoding "
            f"against configs/isruc_s3.yaml before rerunning."
        )
    return np.array([stage_map[v] for v in raw_labels], dtype=np.int64)


def extract_epochs(rec_path: str, labels: np.ndarray, channels: dict, sampling_rate_hz: int):
    # Bypass MNE's extension-based format sniffing by copying bytes to a
    # temp file with a .edf suffix -- the .rec file IS EDF format internally,
    # just named differently.
    with open(rec_path, "rb") as f_in:
        rec_bytes = f_in.read()

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=True) as tmp_file:
        tmp_file.write(rec_bytes)
        tmp_file.flush()
        raw = mne.io.read_raw_edf(tmp_file.name, preload=True, verbose=False)

    if int(round(raw.info["sfreq"])) != sampling_rate_hz:
        raw.resample(sampling_rate_hz)

    ch_names = [channels["eeg"], channels["eog"], channels["emg"]]
    missing = [c for c in ch_names if c not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in {rec_path}. "
            f"Available: {raw.ch_names}. Update configs/isruc_s3.yaml channel names."
        )
    raw.pick(ch_names)

    data = raw.get_data()  # (3, n_samples)
    samples_per_epoch = int(EPOCH_SEC * sampling_rate_hz)
    n_epochs_from_signal = data.shape[1] // samples_per_epoch
    n_epochs = min(n_epochs_from_signal, len(labels))
    if n_epochs_from_signal != len(labels):
        print(f"  [WARN] Signal has {n_epochs_from_signal} epochs but hypnogram "
              f"has {len(labels)} -- using min = {n_epochs}")

    epochs_x = np.stack([
        data[:, i * samples_per_epoch:(i + 1) * samples_per_epoch]
        for i in range(n_epochs)
    ]).astype(np.float32)
    epochs_y = labels[:n_epochs]

    # Lights-off trimming (see module docstring / PAD_MINUTES).
    non_wake_idx = np.where(epochs_y != 0)[0]  # 0 = W in stage_map
    if len(non_wake_idx) > 0:
        lo = max(0, non_wake_idx[0] - PAD_EPOCHS)
        hi = min(n_epochs, non_wake_idx[-1] + PAD_EPOCHS + 1)
        epochs_x = epochs_x[lo:hi]
        epochs_y = epochs_y[lo:hi]

    return epochs_x, epochs_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    os.makedirs(ds["processed_dir"], exist_ok=True)

    stage_map = ds["stage_map"]
    scorer = ds.get("hypnogram_scorer", 1)
    subjects = find_subject_dirs(ds["raw_dir"])
    if args.limit:
        subjects = subjects[: args.limit]

    print(f"Found {len(subjects)} ISRUC-S3 subjects. Using hypnogram scorer {scorer}.")
    manifest = []
    for subject_num in subjects:
        out_path = os.path.join(ds["processed_dir"], f"ISRUC{subject_num}.npz")
        if os.path.exists(out_path):
            print(f"[SKIP] Subject {subject_num} already processed.")
            manifest.append(f"ISRUC{subject_num}")
            continue

        subject_dir = os.path.join(ds["raw_dir"], subject_num)
        rec_path = os.path.join(subject_dir, f"{subject_num}.rec")
        try:
            labels = read_hypnogram(subject_dir, subject_num, scorer, stage_map)
            x, y = extract_epochs(rec_path, labels, ds["channels"], ds["sampling_rate_hz"])
        except Exception as e:
            print(f"[ERROR] Subject {subject_num}: {e}")
            continue

        np.savez_compressed(out_path, x=x, y=y, subject_id=f"ISRUC{subject_num}")
        manifest.append(f"ISRUC{subject_num}")
        counts = np.bincount(y, minlength=5)
        print(f"[OK] ISRUC{subject_num}: {len(y)} epochs "
              f"(W={counts[0]} N1={counts[1]} N2={counts[2]} N3={counts[3]} REM={counts[4]})")

    manifest_path = os.path.join(ds["processed_dir"], "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest))
    print(f"\nDone. {len(manifest)} subjects written to {ds['processed_dir']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()