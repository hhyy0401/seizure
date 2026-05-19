# Project context — LightSTHyper TUSZ reproduction pack

> This file is loaded automatically by Claude Code when running in this
> directory. It gives you (Claude) the project state so the user can jump
> straight into analysis without re-explaining anything.

## What this package is

Reproduction kit for **LightSTHyper**, a seizure detection model on the TUSZ
v2.0.6 dataset. The package contains:

- `src/`: training/evaluation code (subset of a larger research repo, trimmed
  to **window-based detection only** — no point-wise / no prediction loader).
- `ckpts/`: best checkpoints for TUSZ 12s and 60s clip-lengths, 3 seeds each
  (123, 456, 789). These are the runs reported in `RESULTS.md`.
- `RESULTS.md`: final numbers (mean±std across seeds).
- `MODEL.md`: architecture description + analysis pointers.

The owner of this package will use it to **run further analyses**: graph /
hyperedge interpretation, channel attention, ablations on E_h or aux loss,
extensions, etc.

## Model in one paragraph

`light_st_hyper` (in `src/model/light_dyn_hyper.py`, class `LightSTHyper`) is
a 3-stage pipeline on EEG STFT features:

1. **Per-channel uni-directional Mamba backbone** (2 layers, d_model=128).
   Each of the 19 EEG channels is processed independently along time.
2. **Two `SpatioTemporalHyperedgeBlock`s** with `E_h` learnable hyperedges
   (best: E_h=1 for 12s, E_h=3 for 60s). Each hyperedge is a **spatio-temporal
   primitive** — a soft (B,T,N,E_h) membership map pools features over **both
   channel and time** into a single hyperedge embedding, then broadcasts back.
3. **PMA readout** (Set Transformer style, 1 learnable seed) over the channel
   axis, followed by a linear BCE classifier on a single logit.

Loss: BCE with logits. Optionally + per-edge BCE deep supervision (`aux_type
bce`, λ=0.3) on the last hyperedge layer — but the best configs use `aux_type
none`.

## Key configs (best as found in our sweep)

| Setting | TUSZ 12s | TUSZ 60s |
|---|---|---|
| E_h | 1 | 3 |
| aux_type | none | none |
| n_hyper_layers | 2 | 2 |
| Backbone | uni-Mamba (2 layers, d=128, `--no_bidirectional`) | same |
| Node embedding | on (`--use_node_emb`) | same |
| Train batch | 128 | 32 |
| Test batch  | 256 | 64 |
| Epoch cap | 80 | 100 |
| Patience (dev AUROC) | 10 | 10 |
| LR | 1e-3 (Adam) | 1e-3 |
| Weight decay | 5e-4 | 5e-4 |
| Input | FFT magnitudes, 100-dim | same |
| Graph | none (model builds its own hyperedge graph internally) | same |

Test reporting protocol: AUROC at dev-AUROC-best ckpt; F1 at τ\* = F1-best τ on dev.

## What lives where

```
src/main.py                            # entry point (train + test, both detection)
src/args.py                            # all CLI flags
src/utils.py                           # save_dir, checkpoint saver, evaluation helpers
src/data/dataloader_detection.py       # TUSZ loader (FFT, balanced 1:1 train, full dev/test)
src/data/file_markers_detection/       # train/dev/test split (clip-level .h5 paths + labels)
                                       # + normalization stats (mean/std .pkl)
src/model/light_dyn_hyper.py           # LightSTHyper + the SpatioTemporal hyperedge block
src/model/mamba_backbone.py            # BiMambaBackbone (uni-/bi-directional)
src/model/{EvoBrain,DCRNN,...}.py      # baselines (only needed if running comparisons)
```

`ckpts/<tag>/best.pth.tar` is a `torch.save({state_dict, ...})` artifact
produced by `utils.CheckpointSaver`. The matching `args.json` records the
exact training args so you can load the model with the same hyperparams.

## How to load a checkpoint in Python (analysis pattern)

```python
import json, torch, sys
sys.path.insert(0, "src")  # so `from model...` and `from data...` resolve
from model.light_dyn_hyper import LightSTHyper_classification

# pick one checkpoint
ckpt_dir = "ckpts/tusz12_E1_noaux_s789"   # best AUROC seed on 12s
args = json.load(open(f"{ckpt_dir}/args.json"))
import argparse; args_ns = argparse.Namespace(**args)

device = torch.device("cuda")
model = LightSTHyper_classification(args_ns, num_classes=1, device=device).to(device)
state = torch.load(f"{ckpt_dir}/best.pth.tar", map_location=device)
model.load_state_dict(state["model_state_dict"])
model.eval()

# Now you can run a clip through:  logits, H_pool = model(x, seq_lengths)
# Inspect the LAST hyperedge block's membership and hyperedge embedding:
#   model.light_st_hyper.hyper_layers[-1].last_M       # (B, T, N, E_h)
#   model.light_st_hyper.hyper_layers[-1].last_h_edge  # (B, E_h, d_hidden)
```

(Note: the wrapper class is `LightSTHyper_classification`; the inner core is
exposed as `.light_st_hyper` in some versions, otherwise it's just the
module's `forward`. Check `model/light_dyn_hyper.py`.)

## Things the user will likely ask you to do

- **"plot the hyperedge membership for this seizure clip"** → load ckpt,
  forward a chosen clip from `data/file_markers_detection/testSet_seq2seq_12s_sz.txt`,
  pull `last_M` and visualize as heatmap (T × N) per edge.
- **"channel attention topomap"** → look at `pma_readout`'s softmax attention
  over the N channels (visible inside `pma_readout` — easiest is to add a
  hook or copy the code to log the attention).
- **"why does E_h=2 underperform E_h=1 and E_h=3 (non-monotonic)?"** — open
  question, would require training a few more E_h values or looking at how the
  hyperedges partition the channels at convergence.
- **"compare against EvoBrain on the same split"** — train_best.sh has the
  flag pattern; baseline code lives in `src/model/EvoBrain.py` (uses graph
  diffusion + Mamba, different backbone family).

## What's NOT here

- Point-wise (frame-level) detection variants (`light_dense_hyper` etc.) — a
  separate task, separate package.
- Seizure prediction (preictal vs interictal) loader — also a separate task.
- CHB-MIT — the same model trains there too (see `final_chb12.sbatch` in the
  original repo) but the markers + resampled data aren't shipped here.

If the user wants those, they'd need the full `seizure/` directory from the
research repo, not this slim repro pack.

## Sanity check before claiming a number

When the user reports something like "I got AUROC X", verify it came from
the dev-AUROC-best ckpt and that F1 used τ\* tuned on dev (not 0.5). The
`main.py --test` path does this correctly out of the box; if rolling your own
eval loop, replicate that protocol or the numbers will look pessimistic.

## A note on the LightSTHyper backbone

The README/code calls it "uni-Mamba". Concretely: `BiMambaBackbone(...,
bidirectional=False)`. The forward direction is causal (proper Mamba). Best
results in our sweep use this uni-directional variant; the bi-directional
variant is faster to learn but the difference at convergence was small and
the paper-strict ablation prefers uni.
