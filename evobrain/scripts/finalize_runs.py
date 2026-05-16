"""For each completed/SIGTERM'd training run dir under ROOT matching the
target grids, find best.pth.tar + args.json, rebuild model & dataloaders,
load the ckpt, and run evaluate() on dev & test to dump
{dev,test}_results.npz into the same work_dir.

Mirrors main.py's post-training eval chain exactly (dev → τ derived from dev
in our aggregate; here we just dump y_true/y_prob/y_pred).

Usage:
    python scripts/finalize_runs.py <run_dir>
    # or via sbatch array: $RUN_DIR is set from SLURM_ARRAY_TASK_ID
"""
import os, sys, json, glob, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Import from main.py
import main as m
import utils


def load_args_namespace(args_path):
    with open(args_path) as f:
        cfg = json.load(f)
    ns = argparse.Namespace(**cfg)
    # Required defaults that may be missing
    ns.test = True
    if not hasattr(ns, "load_model_path") or ns.load_model_path is None:
        ns.load_model_path = os.path.join(os.path.dirname(args_path), "best.pth.tar")
    return ns


def finalize_one(run_dir):
    # work_dir = inner dir containing args.json + best.pth.tar
    cfgs = glob.glob(os.path.join(run_dir, "**", "args.json"), recursive=True)
    if not cfgs:
        print(f"  SKIP {run_dir}: no args.json"); return False
    work_dir = os.path.dirname(cfgs[0])
    ckpt = os.path.join(work_dir, "best.pth.tar")
    if not os.path.exists(ckpt):
        print(f"  SKIP {run_dir}: no best.pth.tar"); return False

    args = load_args_namespace(cfgs[0])
    # Force save_dir = work_dir so npz lands there
    args.save_dir = work_dir
    args.load_model_path = ckpt

    # Reduce noise
    args.cuda = torch.cuda.is_available()
    device = args.device if args.cuda else "cpu"
    utils.seed_torch(seed=args.rand_seed)

    # cuDNN flag for Conv1d-based models (same as main.py)
    if getattr(args, "model_name", "") in (
            "light_st_hyper_dwsep", "light_st_hyper_timesnet",
            "light_st_hyper_tcn", "ada_mshyper", "mshyper", "st_hyper"):
        torch.backends.cudnn.enabled = False

    # Build dataloaders (use main.py's flow indirectly via duplication is fine)
    print(f"  >> work_dir = {work_dir}")
    print(f"  >> ckpt     = {ckpt}")
    if args.dataset == "CHBMIT":
        from data.dataloader_chb import load_dataset_chb
        dataloaders, _, _ = load_dataset_chb(
            task=args.task, input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size, max_seq_len=args.max_seq_len,
            standardize=False, num_workers=args.num_workers,
            augmentation=False, adj_mat_dir="./data/electrode_graph/adj_mx_3d.pkl",
            graph_type=args.graph_type, top_k=args.top_k,
            filter_type=args.filter_type, use_fft=args.use_fft,
            sampling_ratio=args.sampling_ratio, seed=123,
            preproc_dir=args.preproc_dir,
            return_raw=(args.model_name == "light_mamba_band_plv"))
    else:
        from data.dataloader_detection import load_dataset_detection
        dataloaders, _, _ = load_dataset_detection(
            input_dir=args.input_dir, raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size, max_seq_len=args.max_seq_len,
            standardize=True, num_workers=args.num_workers,
            augmentation=False, adj_mat_dir="./data/electrode_graph/adj_mx_3d.pkl",
            graph_type=args.graph_type, top_k=args.top_k,
            filter_type=args.filter_type, use_fft=args.use_fft,
            sampling_ratio=args.sampling_ratio, seed=123,
            preproc_dir=args.preproc_dir)

    # Build model — reuse logic block from main.py via direct dispatch on model_name
    from model.light_dyn_hyper import LightSTHyper_classification
    backbone_map = {
        "light_st_hyper": "mamba",
        "light_st_hyper_dwsep": "dwsep",
        "light_st_hyper_tcn": "tcn",
        "light_st_hyper_timesnet": "timesnet",
    }
    if args.model_name not in backbone_map:
        print(f"  SKIP: unsupported model_name {args.model_name} for finalize"); return False
    model = LightSTHyper_classification(
        args=args, num_classes=args.num_classes, device=device,
        backbone_type=backbone_map[args.model_name],
        hyper_block_type="static", use_input_norm=False)
    model = utils.load_model_checkpoint(ckpt, model).to(device)

    # log = a minimal stand-in (main.evaluate uses it for prints)
    log = utils.get_logger(work_dir, "finalize")

    print("  >> Evaluating dev...")
    dev_results = m.evaluate(model, dataloaders["dev"], args, work_dir, device,
                             log, is_test=True, nll_meter=None, eval_set="dev")
    log.info(f"DEV: {dev_results}")
    print("  >> Evaluating test...")
    test_results = m.evaluate(model, dataloaders["test"], args, work_dir, device,
                              log, is_test=True, nll_meter=None, eval_set="test",
                              best_thresh=dev_results.get("best_thresh"))
    log.info(f"TEST: {test_results}")
    print(f"  >> npz written: {os.listdir(work_dir)}")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="run dir (parent of TUSZ/...)")
    args = p.parse_args()
    ok = finalize_one(args.run_dir)
    sys.exit(0 if ok else 1)
