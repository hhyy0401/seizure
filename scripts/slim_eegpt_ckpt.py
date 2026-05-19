"""Strip the EEGPT Lightning checkpoint down to just what EEGPTClassifier
(use_predictor=True) actually uses, and optionally cast to fp16.

Pretrained ckpt holds 3 sub-modules from the masked-autoencoder framework:
  - target_encoder.*   (we keep — encoder)
  - predictor.*        (we keep — used at fine-tune when use_predictor=True)
  - reconstructor.*    (we drop — replaced by predictor at fine-tune time)
plus shared tensors like cls_token / pos_embed / time_embed.

Usage:
    python scripts/slim_eegpt_ckpt.py --in big.ckpt --out small.pt [--fp16]
"""
import argparse
import os

import torch

# Pretraining graph has 4 sub-models; fine-tuning (EEGPTClassifier
# use_predictor=True) only needs target_encoder + predictor. The
# online "encoder.*" is the EMA source for target_encoder and not
# touched at fine-tune; "reconstructor.*" is replaced by predictor.
DROP_PREFIXES = ("reconstructor.", "encoder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--fp16", action="store_true",
                    help="cast tensors to fp16 — halves disk size, negligible "
                         "accuracy loss for fine-tuning since head/chan_conv "
                         "stay in fp32 at instantiation.")
    args = ap.parse_args()

    sd = torch.load(args.in_path, map_location="cpu", weights_only=False)
    print(f"loaded {args.in_path}")
    if isinstance(sd, dict):
        print(f"  top-level keys: {list(sd.keys())[:12]}")
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                print(f"  picked sub-dict '{k}' ({len(sd[k])} params)")
                sd = sd[k]
                break

    prefix_count = {}
    for k in sd.keys():
        top = k.split(".", 1)[0] + "."
        prefix_count[top] = prefix_count.get(top, 0) + 1
    print(f"  key prefixes: {prefix_count}")

    dtype = torch.float16 if args.fp16 else torch.float32
    kept, dropped, total_bytes = {}, 0, 0
    for k, v in sd.items():
        if any(k.startswith(p) for p in DROP_PREFIXES):
            dropped += 1
            continue
        if isinstance(v, torch.Tensor):
            kept[k] = v.detach().to(dtype).cpu()
            total_bytes += kept[k].numel() * kept[k].element_size()
        else:
            kept[k] = v
    print(f"  kept {len(kept)} / dropped {dropped} (reconstructor.*)")
    print(f"  expected ~{total_bytes / 1e6:.1f} MB")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(kept, args.out_path)
    out_size = os.path.getsize(args.out_path) / 1e6
    print(f"wrote {args.out_path}  ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
