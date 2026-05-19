"""Dump hyperedge membership M for every test clip (with optional dense labels).

Usage:
    python scripts/dump_membership.py <ckpt_dir> <out_npz> \
        [--raw_data_dir PATH] [--input_dir PATH] [--preproc_dir PATH] \
        [--dense]

ckpt_dir must contain args.json + best.pth.tar (searched recursively).
The --raw_data_dir / --input_dir flags override paths baked into args.json
(useful when running on a different filesystem than the original training).

Saves to out_npz:
    M_first : (S, T, N, E_h) float16  — first hyperedge layer membership
    M_last  : (S, T, N, E_h) float16  — last hyperedge layer membership
    y_true  : (S,) uint8                — clip-level label (max over time if dense)
    y_prob  : (S,) float32              — model sigmoid(logit), clip-level
    dense_y : (S, T) uint8              — per-second label (only if --dense, TUSZ)
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import utils
from tqdm import tqdm


def load_args_namespace(args_path):
    with open(args_path) as f: cfg = json.load(f)
    ns = argparse.Namespace(**cfg)
    ns.test = True
    if not hasattr(ns, "load_model_path") or ns.load_model_path is None:
        ns.load_model_path = os.path.join(os.path.dirname(args_path), "best.pth.tar")
    return ns


def main(ckpt_dir, out_npz, raw_data_dir=None, input_dir=None,
         preproc_dir=None, dense=False):
    cfgs = glob.glob(os.path.join(ckpt_dir, "**", "args.json"), recursive=True)
    if not cfgs: sys.exit(f"no args.json under {ckpt_dir}")
    args_path = cfgs[0]
    args = load_args_namespace(args_path)
    work_dir = os.path.dirname(args_path)
    args.save_dir = work_dir
    args.load_model_path = os.path.join(work_dir, "best.pth.tar")
    args.cuda = torch.cuda.is_available()
    device = args.device if args.cuda else "cpu"
    utils.seed_torch(seed=args.rand_seed)

    # CLI overrides (data paths often differ from training filesystem).
    if raw_data_dir: args.raw_data_dir = raw_data_dir
    if input_dir:    args.input_dir    = input_dir
    if preproc_dir:  args.preproc_dir  = preproc_dir
    if dense:        args.dense_labels = True

    use_dense = bool(dense) and args.dataset != "CHBMIT"

    if args.dataset == "CHBMIT":
        from data.dataloader_chb import load_dataset_chb as load_ds
        dataloaders, _, _ = load_ds(
            task=args.task, input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size, max_seq_len=args.max_seq_len,
            standardize=False, num_workers=args.num_workers,
            augmentation=False, adj_mat_dir="./data/electrode_graph/adj_mx_3d.pkl",
            graph_type=args.graph_type, top_k=args.top_k,
            filter_type=args.filter_type, use_fft=args.use_fft,
            sampling_ratio=args.sampling_ratio, seed=123,
            preproc_dir=args.preproc_dir, return_raw=False)
    else:
        from data.dataloader_detection import load_dataset_detection as load_ds
        dataloaders, _, _ = load_ds(
            input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size, max_seq_len=args.max_seq_len,
            standardize=True, num_workers=args.num_workers,
            augmentation=False, adj_mat_dir="./data/electrode_graph/adj_mx_3d.pkl",
            graph_type=args.graph_type, top_k=args.top_k,
            filter_type=args.filter_type, use_fft=args.use_fft,
            sampling_ratio=args.sampling_ratio, seed=123,
            preproc_dir=args.preproc_dir,
            dense_labels=use_dense)

    from model.light_dyn_hyper import LightSTHyper_classification
    model = LightSTHyper_classification(
        args=args, num_classes=args.num_classes, device=device,
        backbone_type="mamba", hyper_block_type="static", use_input_norm=False)
    model = utils.load_model_checkpoint(args.load_model_path, model).to(device)
    model.eval()

    M_first_list, M_last_list = [], []
    y_true_list, y_prob_list, dense_y_list = [], [], []
    test_loader = dataloaders["test"]
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="dump")):
            if len(batch) == 7:
                x, y, seq_lengths, supports, adj, file_name, raw_signal = batch
            else:
                x, y, seq_lengths, supports, adj, file_name = batch
            x = x.to(device)
            seq_lengths = seq_lengths.view(-1).to(device)
            adj = adj.to(device)

            logits, _ = model(x, seq_lengths, adj)
            core = model.model
            M0 = core.hyper_layers[0].last_M.detach().cpu().to(torch.float16).numpy()
            ML = core.hyper_layers[-1].last_M.detach().cpu().to(torch.float16).numpy()
            M_first_list.append(M0)
            M_last_list.append(ML)
            prob = torch.sigmoid(logits.float()).squeeze(-1).detach().cpu().numpy()
            y_prob_list.append(prob)

            if use_dense:
                # y from dense_labels dataloader is (B, T) per-second labels.
                dy = y.detach().cpu().numpy().astype(np.uint8)
                dense_y_list.append(dy)
                # Clip-level label = any second was seizure.
                y_true_list.append((dy.max(axis=-1) > 0).astype(np.uint8))
            else:
                y_true_list.append(y.view(-1).numpy().astype(np.uint8))

    M_first = np.concatenate(M_first_list, axis=0)
    M_last  = np.concatenate(M_last_list,  axis=0)
    y_prob  = np.concatenate(y_prob_list,  axis=0)
    y_true  = np.concatenate(y_true_list,  axis=0)
    save_kwargs = dict(M_first=M_first, M_last=M_last,
                       y_true=y_true,  y_prob=y_prob)
    if use_dense:
        save_kwargs["dense_y"] = np.concatenate(dense_y_list, axis=0)
        print(f"dense_y {save_kwargs['dense_y'].shape}")
    print(f"M_first {M_first.shape}  M_last {M_last.shape}  "
          f"y_true {y_true.shape}  pos={y_true.sum()}/{y_true.size}")

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **save_kwargs)
    print(f"saved {out_npz}  ({os.path.getsize(out_npz) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir")
    p.add_argument("out_npz")
    p.add_argument("--raw_data_dir", default=None)
    p.add_argument("--input_dir",    default=None)
    p.add_argument("--preproc_dir",  default=None)
    p.add_argument("--dense", action="store_true", default=False,
                   help="Save per-second labels (dense_y); TUSZ only.")
    a = p.parse_args()
    main(a.ckpt_dir, a.out_npz,
         raw_data_dir=a.raw_data_dir, input_dir=a.input_dir,
         preproc_dir=a.preproc_dir, dense=a.dense)
