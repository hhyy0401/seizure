"""Dump hyperedge membership M for every test clip.

Usage:
    python scripts/dump_membership.py <ckpt_dir> <out_npz>

ckpt_dir must contain args.json + best.pth.tar.
Saves to out_npz:
    M_first : (S, T, N, E_h) float16  — first hyperedge layer membership
    M_last  : (S, T, N, E_h) float16  — last hyperedge layer membership
    y_true  : (S,) uint8
    y_prob  : (S,) float32
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


def main(ckpt_dir, out_npz):
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
            preproc_dir=args.preproc_dir)

    from model.light_dyn_hyper import LightSTHyper_classification
    model = LightSTHyper_classification(
        args=args, num_classes=args.num_classes, device=device,
        backbone_type="mamba", hyper_block_type="static", use_input_norm=False)
    model = utils.load_model_checkpoint(args.load_model_path, model).to(device)
    model.eval()

    M_first_list, M_last_list, y_true_list, y_prob_list = [], [], [], []
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
            # LightSTHyper_classification → .model is the actual LightSTHyper
            core = model.model
            M0 = core.hyper_layers[0].last_M.detach().cpu().to(torch.float16).numpy()
            ML = core.hyper_layers[-1].last_M.detach().cpu().to(torch.float16).numpy()
            M_first_list.append(M0)
            M_last_list.append(ML)
            prob = torch.sigmoid(logits.float()).squeeze(-1).detach().cpu().numpy()
            y_prob_list.append(prob)
            y_true_list.append(y.view(-1).numpy().astype(np.uint8))

    M_first = np.concatenate(M_first_list, axis=0)
    M_last = np.concatenate(M_last_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    print(f"M_first {M_first.shape}  M_last {M_last.shape}  y_true {y_true.shape}  pos={y_true.sum()}/{y_true.size}")

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, M_first=M_first, M_last=M_last,
                        y_true=y_true, y_prob=y_prob)
    print(f"saved {out_npz}  ({os.path.getsize(out_npz) / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python dump_membership.py <ckpt_dir> <out_npz>")
    main(sys.argv[1], sys.argv[2])
