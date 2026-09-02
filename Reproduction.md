# DSSNet Reproduction — Spec Sheet (paste this into future chats)

## Goal
Reproduce "DualStabSleepNet (DSSNet): A Dual-Domain Diffusion Stabilization
Network for Robust Sleep Staging" using the **same architecture** described
in the paper, targeting accuracy **equal to or 1-2% above** the paper's
reported numbers — not below. Prior reproduction attempt landed 1-2% below
paper numbers; root-causing that gap (likely subject-split leakage,
under-specified preprocessing, or checkpoint selection) is a priority.

## Workflow convention
- Build/debug code locally (this chat / VS Code), push to GitHub.
- Clone the GitHub repo into a Kaggle notebook and run training there
  (Kaggle GPU: currently unconfirmed quota — free tier assumed unless told
  otherwise).
- One shared codebase, per-dataset YAML configs — never fork the pipeline
  per dataset (this is a common source of silent reproduction drift).

## Data status (update as it changes)
- SleepEDF-78 (PhysioNet Sleep-EDF Database Expanded, full 78-subject set): HAVE
- SleepEDF-20: NOT separately needed — it's subjects SC4000–SC4019 (SC40[00-19]*)
  within the same SleepEDF-78 files. Derived via subject-ID filter, no new download.
- ISRUC-S3: NOT YET OBTAINED — need to source from ISRUC-Sleep dataset site.
- SHHS: DEFERRED. Used in the paper for (a) full within-dataset training AND
  (b) cross-dataset eval of the SleepEDF-78-trained model. Training on SHHS
  is skipped for now; will still need the SHHS *test* split later for the
  cross-dataset row (Table III `DSSNet(ours)*`), not for training.

## Target numbers (paper, Table III / IV)
| Dataset      | ACC  | MF1  | Kappa | N1 acc (hardest stage) |
|--------------|------|------|-------|-------------------------|
| SleepEDF-20  | 89.2 | 85.5 | 0.852 | 62.3% |
| SleepEDF-78  | 88.0 | 84.2 | 0.836 | 62.3% |
| SHHS         | 89.7 | 84.0 | 0.855 | 56.9% |
| ISRUC-S3     | 86.7 | 84.9 | 0.829 | 64.5% |
| SleepEDF-78→SHHS (cross)     | 88.5 | 82.6 | 0.839 | 54.2% |
| SleepEDF-78→ISRUC-S3 (cross) | 85.0 | 83.3 | 0.807 | 61.6% |

## Architecture (must match paper exactly)
1. **Data-domain stabilization**: EDM-preconditioned 1D U-Net denoiser,
   continuous-scale noise σ ∈ [σ_min, σ_max] = [0.05, 0.2], trained with
   scale-weighted MSE (data-prediction parameterization), solve++ deterministic
   sampler at inference, N steps ∈ {8, 12, 16}. Trained independently, frozen
   before downstream use — no classification gradient flows through it.
2. Normalize → STFT (H×W time-frequency) per channel (EEG, EOG, EMG).
3. **ViT backbone**: dim=256, depth=4, heads=4, standard pre-norm transformer
   blocks, patch/flatten embedding of STFT maps, multi-level features F1..F4
   (K=4 hierarchy levels).
4. **Feature-domain stabilization**: EMA teacher (momentum µ=0.999) provides
   stable reference features F_k^t; per-level diffusion module D_ψ^k trained
   to map perturbed features (σ up to σ_feat_max=0.2) back to teacher target;
   loss = scale-weighted MSE per level, λ=0.2 weight vs. classification loss
   in the joint stage. At inference: teacher discarded, feature diffusion
   applied deterministically at σ→0 as a fixed projection.
5. **Two-stage training**: Stage 1 — freeze backbone, train only feature
   diffusion modules near the teacher manifold. Stage 2 — joint fine-tune of
   backbone + fusion head + feature diffusion modules with
   L_total = L_cls + λ Σ_k L_feat^(k).
6. Optimizer AdamW, lr=1e-4 (Table II — no explicit batch size/epochs given
   in the paper; will need to choose sensibly and document the choice since
   this is exactly the kind of unstated detail that causes reproduction gaps).

## Known likely causes of "1-2% below paper" gap (to actively guard against)
- Subject-wise train/val/test split not enforced (leakage across epochs
  from the same subject inflates or deflates depending on which side it's on).
- N1 class imbalance not handled with the same care (paper shows N1 as the
  hardest stage and the biggest reported gain — sampling/weighting matters).
- STFT window/hop parameters not stated in paper — must pick and log exactly,
  then treat as a tunable if gap appears.
- Checkpoint selection: paper likely reports best validation checkpoint, not
  final-epoch — make sure our eval protocol matches (best-val, not last-epoch).
- EMA teacher momentum/warm-up schedule under-specified — implement standard
  EMA warm-up (e.g., ramp µ up over first N steps) rather than fixed µ from
  step 0, since fixed high µ from a randomly-initialized student can stall
  teacher usefulness early in training.

## Bugs found & fixed (log, for continuity)
1. **U-Net channel mismatch** (encoder blocks built for pre-projection
   channel count but run post-projection in forward()) -- fixed, verified
   via shape/backward/sampler test.
2. **U-Net decoder skip-connection off-by-one** (wrong index into
   enc_channels for skip_ch) -- fixed, verified.
3. **Hypnogram mis-pairing** (data correctness bug, caught via a second
   Claude instance reviewing preprocessing output): `find_subject_pairs`
   matched hypnogram files on subject_num alone (`SC400*Hypnogram.edf`),
   which matches BOTH nights of a subject, silently pairing both PSG
   recordings with night 1's hypnogram via `[0]`. Symptom: identical
   epoch counts/class breakdowns for SC4001E0 and SC4002E0. Fixed by
   matching on the full night-specific prefix (`SC4001E`), wildcarding
   only the trailing scorer-code letter (varies: C, J, etc). Verified
   against a synthetic repro of the exact failure.
4. **Subject-split leakage risk**: SC4001E0/SC4002E0 are literally the
   SAME person's two nights. Splitting by recording_id (not true person)
   could put one person's two nights in different splits/folds --
   silent leakage. Fixed: `dataset.py` now groups by
   `person_id_from_recording()` (chars [3:5] of the recording id) before
   splitting; both nights of a person always land in the same split.
   Verified both split and k-fold functions.

5. **Config deep-merge bug**: `train_data_diffusion.py` had its own
   shallow-merge `load_config` (`{**parent, **child}`), which silently
   wiped out sibling keys (e.g. overriding `optim.batch_size` deleted
   `optim.lr`). Consolidated into one correct recursive `deep_merge` in
   `src/config_utils.py`, used by preprocessing + both trainers. Verified.
6. **`torch.load` weights_only default** (PyTorch 2.6 changed the default
   to `True`, which rejects numpy arrays stored in our checkpoints,
   e.g. norm mean/std). Fixed with explicit `weights_only=False` on both
   checkpoint loads (our own trusted checkpoints only).

## Current session progress
- Repo scaffold, config system (now single correct loader), SleepEDF-78/20
  preprocessing (bug-fixed + integration-verified).
- Full model stack built and unit-verified: data-domain diffusion, STFT,
  ViT backbone, feature-domain diffusion, DSSNet assembly.
- Dataset/DataLoader: person-level subject-wise split, train-only
  normalization, N1-balanced sampling -- verified.
- train_data_diffusion.py (Stage 0), train_dssnet_two_stage.py (Stage 1+2),
  eval/metrics.py -- ALL THREE built and CLI-integration-tested end-to-end
  on synthetic data (full run: preprocessing format -> Stage0 checkpoint ->
  Stage1 -> Stage2 -> best-checkpoint reload -> final test metrics, no
  crashes). Not yet run on real SleepEDF data or at real paper-scale
  hyperparameters (n_steps=8/12/16, full epoch counts) -- that's the next
  actual milestone, on your Kaggle GPU.
- STILL TODO: ISRUC-S3 preprocessing, Kaggle notebook wiring, running for
  real on your data and comparing against Table III/IV target numbers.