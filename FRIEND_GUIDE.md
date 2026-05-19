# 친구용 — TUSZ / CHB-MIT seizure detection (baselines + LightSTHyper)

This repo lets you reproduce LightSTHyper + paper baselines on
**TUSZ** (12s & 60s) and **CHB-MIT** (12s) seizure detection.
The CHB-MIT data split is **frozen and committed** in
[src/data/file_markers_chb/](src/data/file_markers_chb/) — no need to
regenerate it; just pull the repo and the split is bit-exact.

**What works out of the box:**

| Task | TUSZ | CHB-MIT |
|---|---|---|
| Window detection (clip-level) | ✓ all models | ✓ all models |
| Pointwise detection (frame-level) | ✓ `*_dense` neural models | ✗ not wired |
| Prediction (preictal vs interictal) | ✓ neural models (needs prediction markers) | ✗ not wired |

**Models available:**
- Neural: `light_st_hyper` (ours), `evobrain`, `dcrnn`, `evolvegcn`, `graphs4mer`, `gru_gcn`, `lstm`, `cnnlstm`, `BIOT`, plus all `*_dense` variants for pointwise.
- Classical: `svm`, `rf` (Random Forest) — see [§ Classical baselines](#classical-baselines-svm--random-forest) below.

---

## TL;DR

```bash
# 1. CHB-MIT raw data + resampled .h5 already prepared by you (see "Data prep")
# 2. Activate the conda env (mamba_ssm + torch_geometric required)
# 3. Submit:

sbatch --array=0-2 --export=ALL,MODEL=evolvegcn,CLIP_LEN=12  sbatch/train/baseline_chb.sbatch
sbatch --array=0-2 --export=ALL,MODEL=graphs4mer,CLIP_LEN=12 sbatch/train/baseline_chb.sbatch
```

3-seed arrays (123 / 456 / 789). Each job ≈ 1.5–3h on A100.
Test AUROC/F1 are dumped to `$SAVE_DIR/test_results.npz`.

---

## What's added compared to upstream LightSTHyper

| File | Purpose |
|---|---|
| [src/model/EGCN.py](src/model/EGCN.py) | EvolveGCN-O (verbatim port of IBM/EvolveGCN `egcn_o.py` + classification wrapper) |
| [src/model/graphs4mer.py](src/model/graphs4mer.py) | GraphS4mer (faithful port of tsy935/graphs4mer; uses Mamba in place of S4 — see docstring for why) |
| [src/main.py](src/main.py) | dispatches `--model_name evolvegcn` / `--model_name graphs4mer` / `lstm` / `BIOT` / `dcrnn` ... |
| [sbatch/train/baseline_chb.sbatch](sbatch/train/baseline_chb.sbatch) | CHB runner — evolvegcn / graphs4mer |
| [sbatch/train/baseline_chb_lbd.sbatch](sbatch/train/baseline_chb_lbd.sbatch) | CHB runner — lstm / cnnlstm / BIOT / dcrnn / gru_gcn |
| [sbatch/train/baseline_tusz.sbatch](sbatch/train/baseline_tusz.sbatch) | TUSZ counterpart (evolvegcn / graphs4mer) |
| [sbatch/train/baseline_tusz_lbd.sbatch](sbatch/train/baseline_tusz_lbd.sbatch) | TUSZ runner — lstm / cnnlstm / BIOT / dcrnn / gru_gcn |
| [sbatch/train/baseline_foundation.sbatch](sbatch/train/baseline_foundation.sbatch) | TUSZ + CHB runner — labram / eegpt (see § EEG foundation-model baselines) |
| [src/run_classical.py](src/run_classical.py) | Classical-ML baseline trainer: SVM / Random Forest on FFT features (clip-level only) |
| [sbatch/train/baseline_classical.sbatch](sbatch/train/baseline_classical.sbatch) | Runner for the classical baselines (works for TUSZ + CHB, 12s + 60s) |

Read the docstrings in the model files — they explain the API and
the deviations from the original papers.

---

## File pointers (for Claude / cursor when navigating)

If you (the friend) open this in Claude Code, the relevant files are:

- **Models you're training:**
  - [src/model/EGCN.py](src/model/EGCN.py) (EvolveGCN-O)
  - [src/model/graphs4mer.py](src/model/graphs4mer.py) (GraphS4mer, Mamba-temporal)
- **Training entry point:** [src/main.py](src/main.py) — search for `"evolvegcn"` / `"graphs4mer"` to see how forward signatures are wired.
- **CHB-MIT data loader:** [src/data/dataloader_chb.py](src/data/dataloader_chb.py)
- **CHB-MIT data split (frozen):** [src/data/file_markers_chb/](src/data/file_markers_chb/) — `trainSet_seq2seq_12s_{sz,nosz}.txt`, `devSet_*`, `testSet_*`
- **Channel list / freq:** [src/data/constants_chb.py](src/data/constants_chb.py)
- **All training/eval flags:** [src/args.py](src/args.py)
- **Sbatch:** [sbatch/train/baseline_chb.sbatch](sbatch/train/baseline_chb.sbatch) (read this first to see the exact CLI invocation)

---

## Data prep prerequisites

Before running, you (the friend) need three things on disk:

1. **CHB-MIT raw EDFs**, layout:
   ```
   $RAW_DIR/chb01/{chb01-summary.txt, chb01_01.edf, chb01_02.edf, ...}
   $RAW_DIR/chb02/...
   ...up to chb24
   ```
   (Download from https://physionet.org/content/chbmit/1.0.0/)

2. **Resampled .h5 files at 200 Hz** (one per EDF), in `$RESAMPLED_DIR`:
   ```
   $RESAMPLED_DIR/chb01_01.h5  # contains dataset 'resampled_signal' shape (18, n_samples)
   ...
   ```
   Use [src/data/preprocess_chb.py](src/data/preprocess_chb.py) (or its sibling
   [src/data/resample_signals.py](src/data/resample_signals.py)) to produce these.

3. **Pull this repo** — `src/data/file_markers_chb/*.txt` are already in git,
   so your train/dev/test split is guaranteed identical to ours (see
   "Data split" below).

The conda env needs: `torch`, `torch_geometric`, `mamba_ssm` (for both
baselines), `h5py`, `pyedflib`, `numpy<2`, plus the usual sklearn / scipy /
networkx / tensorboardX.

---

## Data split — how it was made + how to verify

The CHB-MIT split is a per-patient random 70/15/15 of clip indices, seed
123, generated by [src/data/build_file_markers_chb.py](src/data/build_file_markers_chb.py).
The output is the txt files in [src/data/file_markers_chb/](src/data/file_markers_chb/).

**You do not need to regenerate the split.** The txt files are committed to
git so cloning the repo gives you the exact same split. Just verify after
clone:

```bash
wc -l src/data/file_markers_chb/*.txt
# match counts against what hkim3239 reports
```

If for any reason you want to regenerate (e.g., different patient subset),
run with the same seed:
```bash
python src/data/build_file_markers_chb.py \
    --raw_dir   $RAW_DIR \
    --input_dir $RESAMPLED_DIR \
    --out_dir   src/data/file_markers_chb \
    --clip_len  12 \
    --seed      123
```
This is deterministic given the same numpy version (>=1.17 for
`default_rng`) and the same set of EDFs present in `$RAW_DIR`.

---

## Running

```bash
# Edit paths in sbatch/train/baseline_chb.sbatch (lines for ENV_PATH, RAW, RESAMPLED)
# to point to your machine, then:

sbatch --array=0-2 --export=ALL,MODEL=evolvegcn,CLIP_LEN=12  sbatch/train/baseline_chb.sbatch
sbatch --array=0-2 --export=ALL,MODEL=graphs4mer,CLIP_LEN=12 sbatch/train/baseline_chb.sbatch
```

Each invocation submits a 3-seed array (123/456/789). Total: 6 jobs.

Override hyperparams via the same `--export=ALL,...` mechanism, e.g.:
```bash
sbatch --array=0-2 --export=ALL,MODEL=graphs4mer,CLIP_LEN=12,LR=3e-4,NUM_EPOCHS=60 \
    sbatch/train/baseline_chb.sbatch
```

---

## Reporting

Test metrics follow the package's protocol: dev-AUROC-best checkpoint, F1
at τ\* tuned on dev (NOT 0.5). The `main.py --test` path implements this
out of the box, no manual eval loop needed.

Per-job output (e.g. `runs/graphs4mer_chb12_s123_<jobid>/`):
- `test_results.npz` — y_true / y_prob / file_names
- `args.json` — exact hyperparams
- `best.pth.tar` — best dev-AUROC checkpoint
- `train.log` — full log

For multi-seed mean±std, aggregate `test_results.npz` across the 3 seeds.

---

## Sanity check before claiming numbers

Same protocol as the rest of this repo: dev-AUROC-best ckpt, F1 at τ\*
tuned on dev. Don't report F1@0.5.

---

## Classical baselines (SVM / Random Forest)

**File:** [src/run_classical.py](src/run_classical.py) (standalone — does
not go through `main.py`).
**Sbatch:** [sbatch/train/baseline_classical.sbatch](sbatch/train/baseline_classical.sbatch)

**Pipeline:** Loads (T, N, F)-shaped log-FFT clips through the **same
dataloader the neural baselines use** (CHB-MIT via `dataloader_chb.py`,
TUSZ via `dataloader_detection.py`, `use_fft=True`), then reduces to a
per-clip feature vector. Fits sklearn's `SVC(kernel='rbf')` /
`RandomForestClassifier`, evaluates with **dev-AUROC-best test AUROC** and
**F1 at τ\* tuned on dev** — exact same reporting protocol as the neural runs.

**Feature reduction modes** (`--feature_mode`):
- `mean` (recommended): mean log-FFT over time → (N · F)-dim per clip
  (≈ 1900 for TUSZ-19ch, ≈ 2200 for CHB-22ch). Defendable, matches what
  the neural baselines see, statistically sound given small balanced train set.
- `flatten`: full flatten of (T, N, F) → up to 26,400-dim. More info but
  curse of dimensionality and slow SVC.

**Hyperparameter modes** (`--vanilla`):
- default: `class_weight='balanced'`, RF `n_estimators=200`.
- `--vanilla`: `class_weight=None`, RF `n_estimators=100`. Matches sklearn
  defaults; closest plausible approximation to the published
  EvoBrain-Table-1 SVM/RF setup (their spec is not released).

**Examples:**
```bash
# CHB-MIT 12s detection, SVM, 3 seeds (vanilla mean, default for paper-style reporting)
for SEED in 123 456 789; do
  sbatch --export=ALL,DATASET=CHBMIT,MODEL=svm,FEAT=mean,SEED=$SEED,CLIP_LEN=12,VANILLA=1 \
      sbatch/train/baseline_classical.sbatch
done

# TUSZ 60s detection, Random Forest, 3 seeds (vanilla mean)
for SEED in 123 456 789; do
  sbatch --export=ALL,DATASET=TUSZ,MODEL=rf,FEAT=mean,SEED=$SEED,CLIP_LEN=60,VANILLA=1 \
      sbatch/train/baseline_classical.sbatch
done

# Compare to the strongly-regularized "balanced" variant (drop VANILLA=1)
sbatch --export=ALL,DATASET=TUSZ,MODEL=rf,FEAT=flatten,SEED=123,CLIP_LEN=12 \
    sbatch/train/baseline_classical.sbatch
```

**Output:** `runs/<model>_<dataset>12_<feat>[_vanilla]_s<seed>_<jobid>/results.json`
contains `test_AUROC`, `test_F1`, `tau_star_on_dev`, etc.

**Caveats:**
- SVM/RF are inherently **clip-level**. Pointwise / frame-level prediction
  is not implemented — would need one classifier per timestep or a
  rolling window adaptation.
- Prediction (preictal vs interictal) is not implemented in
  `run_classical.py` either — only `--task detection`. Easy to extend
  by adding `from data.dataloader_prediction import load_dataset_prediction`
  and dispatching on `--task`.

---

## Pointwise (frame-level) detection — TUSZ only

The repo has per-second `_dense` variants for the neural models. CHB-MIT
**is not wired up** for pointwise (the CHB loader explicitly raises
NotImplementedError when `dense_labels=True`).

**Available `*_dense` models** (set via `--model_name`):
- `light_dense_hyper` (ours, per-t HyperedgeBlock)
- `evobrain_dense`, `dcrnn_dense`, `gru_gcn_dense`, `lstm_dense`,
  `cnnlstm_dense`, `biot_dense`

All `*_dense` models **require** `--dense_labels`. The dataloader then
returns per-second labels y of shape (T,), the model emits (B, T, 1)
logits, trainer does BCE over (B, T).

**Smoothness regularizer (optional):**
`--smooth_weight λ` adds an L2 penalty on adjacent-timestep logit
differences, encoding the prior that seizures are continuous in time:
loss = BCE + λ · mean((logits[:, t+1] − logits[:, t])²)

**Example (TUSZ 60s, our `light_dense_hyper`, 3 seeds):**
```bash
for SEED in 123 456 789; do
  python src/main.py \
      --dataset TUSZ --task detection --model_name light_dense_hyper \
      --dense_labels --max_seq_len 60 --time_step_size 1 \
      --num_nodes 19 \
      --raw_data_dir $TUSZ_RAW --input_dir $TUSZ_RESAMPLED \
      --preproc_dir $TUSZ_PREPROC_60s \
      --use_fft --graph_type none \
      --rand_seed $SEED --num_epochs 100 --patience 15 \
      --train_batch_size 32 --test_batch_size 64 \
      --lr_init 1e-3 --l2_wd 5e-4 \
      --metric_name auroc --eval_every 2 --skip_midtest \
      --smooth_weight 0.05 \
      --save_dir runs/light_dense_hyper_tusz60_s${SEED}
done
```

Swap `--model_name` to use a baseline (`evobrain_dense`, `lstm_dense`,
`biot_dense`, etc.). Reporting is per-frame AUROC + F1@τ\* (computed
automatically by `main.py --test` path).

---

## Prediction (preictal vs interictal) — TUSZ only

Loader exists ([src/data/dataloader_prediction.py](src/data/dataloader_prediction.py))
but **prediction markers (`src/data/file_markers_prediction/`) are NOT
committed**. You need to generate them (analogous to
[src/data/build_file_markers_v206.py](src/data/build_file_markers_v206.py)
for detection) or copy them in from your prior pipeline.

The CHB-MIT loader explicitly blocks prediction with NotImplementedError —
needs a port (parse preictal windows from `chbXX-summary.txt`, build
markers, expose the task in `dataloader_chb.py`'s entry point).

Once `file_markers_prediction/` is in place:
```bash
python src/main.py \
    --dataset TUSZ --task prediction --model_name lstm \
    --max_seq_len 12 --num_nodes 19 \
    --raw_data_dir $TUSZ_RAW --input_dir $TUSZ_RESAMPLED \
    --use_fft --graph_type none \
    --train_batch_size 128 --test_batch_size 256 \
    --num_epochs 80 --patience 10 --lr_init 1e-3 \
    --rand_seed 123 --metric_name auroc --skip_midtest --eval_every 2 \
    --save_dir runs/lstm_pred_tusz12_s123
```

Same pattern works for `BIOT`, `dcrnn`, `evolvegcn`, etc. (binary clip-
level classifiers).

---

## EEG foundation-model baselines (LaBraM + EEGPT)

To match the EvoBrain (NeurIPS 2025) Table 1 foundation-model column we
ship end-to-end fine-tuning of two pretrained encoders:

| Model | Source | Pretrained weight |
|---|---|---|
| **LaBraM** (Jiang et al., ICLR 2024) | `braindecode.models.Labram` (re-keyed version of `935963004/LaBraM`) | `braindecode_labram_base.pt` (23 MB) — pulled once from HuggingFace |
| **EEGPT** (Wang et al., NeurIPS 2024) | Vendored from `BINE022/EEGPT` (figshare 25866970) | `eegpt_mcae_58chs_4s_large4E.ckpt` (974 MB) |

Both weights live on scratch (the project repo only holds code + light ckpts):

```
/storage/scratch1/3/hkim3239/eeg/pretrained/labram/braindecode_labram_base.pt
/storage/scratch1/3/hkim3239/eeg/pretrained/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt
```

### Files

| File | Purpose |
|---|---|
| [src/model/labram.py](src/model/labram.py) | `LaBraM_classification` wrapper. Uses `braindecode.models.Labram`; loads ckpt with `strict=False`; reshapes our dataloader's `(B, T, N, 200)` to `(B, N, T*200)`. |
| [src/model/eegpt.py](src/model/eegpt.py) | `EEGPT_classification` wrapper. Builds `EEGPTClassifier` with `use_chan_conv=True` so any channel montage is projected to the pretrained 58-channel space; lets the model's internal `temporal_interpolation` resample our 12/60-s clips to the pretrained 4-s window. |
| [src/model/eegpt_pretrained/](src/model/eegpt_pretrained/) | Vendored encoder + classifier code from the EEGPT figshare share (verbatim). |
| [sbatch/train/baseline_foundation.sbatch](sbatch/train/baseline_foundation.sbatch) | `MODEL=labram\|eegpt × DATASET=TUSZ\|CHBMIT × CLIP_LEN=12\|60`. Uses lr=5e-4, weight_decay=0.05, smaller batches than baseline runners. |

### Run

```bash
# TUSZ 12s — both foundation models, 3 seeds each
sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=TUSZ,CLIP_LEN=12   sbatch/train/baseline_foundation.sbatch
sbatch --array=0-2 --export=ALL,MODEL=eegpt,DATASET=TUSZ,CLIP_LEN=12    sbatch/train/baseline_foundation.sbatch
# TUSZ 60s
sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=TUSZ,CLIP_LEN=60   sbatch/train/baseline_foundation.sbatch
sbatch --array=0-2 --export=ALL,MODEL=eegpt,DATASET=TUSZ,CLIP_LEN=60    sbatch/train/baseline_foundation.sbatch
# CHB-MIT 12s
sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=CHBMIT,CLIP_LEN=12 sbatch/train/baseline_foundation.sbatch
sbatch --array=0-2 --export=ALL,MODEL=eegpt,DATASET=CHBMIT,CLIP_LEN=12  sbatch/train/baseline_foundation.sbatch
```

### Caveats worth flagging in the paper

- **EEGPT context length.** EEGPT was pretrained on 4-second windows. We feed 12/60-s clips and the model's built-in `temporal_interpolation` downsamples to 4 s, which loses temporal resolution. This is the same protocol the EvoBrain paper used (their reported numbers: AUROC 0.803/0.743 on TUSZ 12s/60s).
- **CHB-MIT bipolar montage.** Both pretrained encoders expect referential channels. For CHB-MIT we keep the bipolar `X-Y` channels and rely on `use_chan_conv=True` to learn a projection — this is the same trade-off BIOT makes for cross-montage transfer. Numbers may be slightly pessimistic vs. a referential CHB-MIT setup.
- **Pretrained weights ship in the repo** under [ckpts/pretrained/](ckpts/pretrained/):
  - `labram_base.pt` — 23 MB, braindecode-keyed mirror of `935963004/LaBraM`-base.
  - `eegpt_base_slim_fp16.pt` — 98 MB, slimmed from `eegpt_mcae_58chs_4s_large4E.ckpt` (974 MB) by dropping the pretraining-only `encoder.` / `reconstructor.` sub-modules and casting to fp16. Slim → fits under GitHub's 100 MB single-file limit. Numerically equivalent to the original ckpt for fine-tuning (target_encoder + predictor are kept verbatim).
  - `src/model/{labram,eegpt}.py` resolves these paths automatically via `__file__`, so a fresh `git clone` runs out of the box.
  - [scripts/download_pretrained.sh](scripts/download_pretrained.sh) is a fallback only — when the ckpts/ tree is missing (sparse checkout) or you want the original 974 MB Lightning ckpt to re-derive the slim file via [sbatch/figures/slim_eegpt.sbatch](sbatch/figures/slim_eegpt.sbatch).

### One-shot setup (the friend's first time)

```bash
git clone <repo> && cd seizure
pip install -r requirements.txt          # pulls braindecode==1.3.2 etc.

# done — pretrained weights already in ckpts/pretrained/
sbatch --array=0-2 --export=ALL,MODEL=labram,DATASET=TUSZ,CLIP_LEN=12 \
       sbatch/train/baseline_foundation.sbatch
```

The LightSTHyper "main result" ckpts (`ckpts/tusz12_E1_noaux_s{123,456,789}/`,
`ckpts/tusz60_E3_noaux_s{123,456,789}/`) **are shipped in git** (3.3 MB each)
so the friend can `--test` directly without re-training.

### Env note (don't bump braindecode)

`pip install braindecode` (no version pin) currently pulls 1.5+, which
requires torch>=2.10 — that upgrade breaks `mamba-ssm==2.2.4` (used by
`light_st_hyper`'s Mamba backbone) and `libtorch_global_deps.so` ends up
missing. **Pin to `braindecode==1.3.2`** (the latest 1.x that still
accepts torch 2.5.1). See [requirements.txt](requirements.txt).



---

## LightSTHyper + Temporal Attention (current best on TUSZ 60s)

After the LightSTHyper sweep (Round 1-3 on `s123`, then 3-seed verification on
`s456`/`s789`) we found a small architectural addition that consistently helps
on **TUSZ 60s** — a per-channel temporal self-attention layer wedged between
the hyperedge stack and the PMA readout. We call it `tattn` (Temporal
Attention) and it's a CLI flag now.

### What `tattn` is

A per-channel `T × T` multi-head attention block (`nn.MultiheadAttention`,
4 heads, d_hidden=d_model=128, residual + LayerNorm) applied to the
post-hyperedge tensor `(B, T, N, d_hidden)`. For each channel `n` the
timesteps attend to each other bidirectionally — exactly the long-range
temporal mixing that **uni-directional Mamba** can't produce (Mamba is
causal, so timestep `t` only sees `≤ t`).

Code: [src/model/light_dyn_hyper.py:TemporalAttentionLayer](src/model/light_dyn_hyper.py)

Insertion point (default `tattn_position=after`):

```
x → per-channel uni-Mamba → 2× SpatioTemporalHyperedge → ★tattn★ → mean(T) → PMA(N) → FC
```

The hyperedge `M` membership map is still produced as before (so the
interpretability story is preserved); `tattn` adds ~100 k params (d² × 4
projections) and ~5 % wall-time overhead.

### Best TUSZ 60s config (3-seed test, beats GRU-GCN 0.907/0.640 on AUROC)

| Seed | Test AUROC | Test F1 |
|---|---|---|
| s123 | 0.907 | 0.560 |
| s456 | 0.913 | 0.664 |
| s789 | 0.905 | 0.618 |
| **mean ± std** | **0.908 ± 0.004** | **0.614 ± 0.052** |

Hyperparameters that won the sweep:

| Flag | Value | Notes |
|---|---|---|
| `--model_name` | `light_st_hyper` | unchanged |
| `--max_seq_len` | `60` | 60-s clips, `--time_step_size 1` |
| `--n_hyperedges` | `3` | Round 2 winner. {1,2,3} swept; 3 best on 60s. |
| `--n_hyper_layers` | `2` | hL3 was tested too — slight gain only with 3 seeds, not worth complexity. |
| `--num_rnn_layers` | `2` | Mamba depth (default). |
| `--rnn_units` | `128` | d_model = d_hidden. |
| `--no_bidirectional` |  | uni-Mamba (paper-strict). |
| `--use_node_emb` |  |  |
| `--use_fft` |  | log-FFT input. |
| `--graph_type` | `none` | LightSTHyper builds its own hyperedge graph; no external adj. |
| `--aux_type` | `none` | **aux loss DISABLED** — both `bce` and `entropy` lost in Round 1. |
| `--dropout` | `0.2` | Round 1 winner (vs 0.0 default). |
| `--l2_wd` | `5e-4` |  |
| `--lr_init` | `1e-3` | Adam. |
| `--max_grad_norm` | `5.0` |  |
| `--n_pma_seeds` | `1` |  |
| `--temporal_attn` |  | **new flag — enables `tattn`.** |
| `--temporal_attn_heads` | `4` | (default) |
| `--tattn_n_layers` | `1` | one tattn layer is enough; `2` plateaued at the same dev. |
| `--tattn_position` | `after` | (default — between hyperedge stack and PMA) |
| `--tattn_pos_enc` | _off_ | learnable T-positional embedding — tried, no gain. |
| `--tattn_causal` | _off_ | causal mask hurt — kept bidirectional. |
| `--use_fft_mixer` | _off_ | FFT mixer variant — tried, didn't help. |
| `--readout_concat` | _off_ | raw-feature skip — tried, marginal. |
| `--train_batch_size` | `32` |  |
| `--test_batch_size` | `64` |  |
| `--num_epochs` | `40` | cap (best epoch was ~12–25). Original full-train used `100`. |
| `--patience` | `10` |  |
| `--eval_every` | `3` | every 3 epochs. |
| `--data_augment` |  | random-reflect + random-scale (train-only). |

### Reference sbatch — full Round 2/3 invocation

The sweep we used (`sbatch/train/sweep_tusz60_phase3.sbatch`, task 0 =
`tattn_s456` in the array) shows the exact command. Minimal stand-alone
invocation:

```bash
python src/main.py \
    --dataset TUSZ --task detection --model_name light_st_hyper \
    --num_nodes 19 --max_seq_len 60 --time_step_size 1 \
    --raw_data_dir $TUSZ_RAW --input_dir $TUSZ_RESAMPLED \
    --graph_type none --use_fft --use_node_emb --no_bidirectional \
    --rnn_units 128 --num_rnn_layers 2 \
    --n_hyper_layers 2 --n_hyperedges 3 --n_pma_seeds 1 \
    --aux_type none \
    --dropout 0.2 --l2_wd 5e-4 --max_grad_norm 5.0 \
    --num_epochs 40 --patience 10 --eval_every 3 \
    --train_batch_size 32 --test_batch_size 64 \
    --lr_init 1e-3 --num_workers 8 \
    --rand_seed 123 --metric_name auroc --data_augment \
    --temporal_attn --temporal_attn_heads 4 --tattn_n_layers 1 \
    --save_dir runs/lst_tattn_tusz60_s123
```

For the published 3-seed mean, swap `--rand_seed` to `123`/`456`/`789`.

### Notes for the friend

- **aux loss is off.** Round 1 swept `aux_type ∈ {none, bce, entropy}`;
  `none` was best on 60s. The `--aux_type none` flag is therefore the
  default for all the winning configs.
- **Two `attention` modules, different axes.** `tattn` mixes the
  time axis; PMA pools the channel axis. They are complementary, not
  redundant. Keep them as separate forward stages in code (different
  tensor shapes, different roles) but you can group them conceptually
  as "two attention components" in writeups.
- **F1 has higher variance than AUROC** (std 0.052 vs 0.004 across
  s123/s456/s789). Report both with std; reviewers will ask.
- **CLI flags new in this branch** (all default to off):
  `--temporal_attn`, `--temporal_attn_heads`, `--tattn_n_layers`,
  `--tattn_pos_enc`, `--tattn_position`, `--tattn_causal`,
  `--use_fft_mixer`, `--readout_concat`.

