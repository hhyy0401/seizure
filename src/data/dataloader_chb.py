"""
CHB-MIT seizure detection dataloader for EvoBrain.

Mirrors data/dataloader_detection.py (TUSZ) but adapted for CHB-MIT:
  - 18 bipolar channels (constants_chb.INCLUDED_CHANNELS_CHB) instead of 19
    referential channels.
  - Seizure annotations parsed from per-patient summary.txt files (not TSE).
  - File walker strictly matches '.edf' suffix to avoid '.edf.seizures'.
  - Distance-based combined graph is not available (3D adjacency is for
    referential montage). Use graph_type='individual' or 'dynamic'.
"""
import os
import sys
import pickle
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# repo root on sys.path for top-level imports like `utils`, `constants`
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import utils  # noqa: E402
from utils import StandardScaler  # noqa: E402
from data.data_utils import comp_xcorr, keep_topk, computeFFT  # noqa: E402
from data.constants_chb import INCLUDED_CHANNELS_CHB, FREQUENCY  # noqa: E402

FILEMARKER_DIR = _THIS.parent / "file_markers_chb"


# ----------------------------- helpers -----------------------------


def _time_to_seconds(text: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(f"Cannot parse seconds from: {text!r}")
    return float(m.group(1))


def parse_summary_seizures(summary_path: Path):
    """Returns {edf_basename_without_ext: [(start_s, end_s), ...]}."""
    out = {}
    current = None
    pending_starts = {}
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("File Name:"):
                current = line.split(":", 1)[1].strip()
                if current.endswith(".edf"):
                    current = current[:-4]
                out[current] = []
                pending_starts = {}
                continue
            if current is None:
                continue
            m = re.match(r"Seizure(?:\s+(\d+))?\s+Start Time\s*:\s*(.+)$", line)
            if m:
                pending_starts[m.group(1)] = _time_to_seconds(m.group(2))
                continue
            m = re.match(r"Seizure(?:\s+(\d+))?\s+End Time\s*:\s*(.+)$", line)
            if m:
                idx = m.group(1)
                end_s = _time_to_seconds(m.group(2))
                if idx in pending_starts:
                    out[current].append((pending_starts.pop(idx), end_s))
    return out


def build_seizure_index(raw_data_dir: str):
    """Walk raw_data_dir/chbXX/chbXX-summary.txt files and build a map
    edf_basename -> list of (start_s, end_s)."""
    seizure_map = {}
    raw_root = Path(raw_data_dir)
    for sub in sorted(raw_root.iterdir()):
        if not sub.is_dir():
            continue
        for summary in sub.glob("*-summary.txt"):
            seizure_map.update(parse_summary_seizures(summary))
    return seizure_map


def list_edf_files(raw_data_dir: str):
    """Strictly match '.edf' suffix; excludes '.edf.seizures'."""
    out = []
    for path, _, files in os.walk(raw_data_dir):
        for name in files:
            if name.endswith(".edf"):
                out.append(os.path.join(path, name))
    return out


def _compute_slice(h5_fn, edf_basename, clip_idx, seizure_map,
                   time_step_size=1, clip_len=12, is_fft=False,
                   return_raw=False):
    """Read one clip from a resampled h5 and return (eeg_clip, label[, raw_clip]).

    eeg_clip shape: (num_time_steps, num_channels, time_step_size*FREQUENCY)
    raw_clip shape: (num_channels, clip_len*FREQUENCY) — only when return_raw.
    """
    physical_clip_len = int(FREQUENCY * clip_len)
    physical_time_step = int(FREQUENCY * time_step_size)
    start_w = clip_idx * physical_clip_len
    end_w = start_w + physical_clip_len

    # Lazy slice read — only fetch the 12s window we need rather than the
    # entire recording (originally `f["resampled_signal"][()]`, which forced
    # the full multi-GB array into memory per __getitem__ and was the #1
    # bottleneck of CHB-MIT data loading).
    with h5py.File(h5_fn, "r") as f:
        curr = f["resampled_signal"][:, start_w:end_w]

    sz_windows = seizure_map.get(edf_basename, [])

    raw_clip = curr.astype(np.float32) if return_raw else None

    steps = []
    s = 0
    while s <= curr.shape[1] - physical_time_step:
        e = s + physical_time_step
        x = curr[:, s:e]
        if is_fft:
            x, _ = computeFFT(x, n=physical_time_step)
        steps.append(x)
        s = e
    eeg_clip = np.stack(steps, axis=0)

    is_sz = 0
    # convert window [start_w, end_w) in samples back to seconds for clarity
    start_s_sec = start_w / FREQUENCY
    end_s_sec = end_w / FREQUENCY
    for s_t, e_t in sz_windows:
        if not (end_s_sec <= s_t or start_s_sec >= e_t):
            is_sz = 1
            break
    if return_raw:
        return eeg_clip, is_sz, raw_clip
    return eeg_clip, is_sz


def parseTxtFiles(split_type, seizure_file, nonseizure_file,
                  cv_seed=123, scale_ratio=1):
    """Same balancing semantics as TUSZ dataloader."""
    np.random.seed(cv_seed)
    with open(seizure_file, "r") as f:
        sz_str = f.readlines()
    with open(nonseizure_file, "r") as f:
        nosz_str = f.readlines()

    if split_type == "train" and len(sz_str) > 0:
        num = int(scale_ratio * len(sz_str))
        sz_idx = np.arange(len(sz_str))
        np.random.shuffle(sz_idx)
        sz_str = [sz_str[i] for i in sz_idx[:num]]
        np.random.shuffle(nosz_str)
        nosz_str = nosz_str[:num]

    combined = sz_str + nosz_str
    np.random.shuffle(combined)
    tuples = []
    for line in combined:
        a, b = line.strip("\n").split(",")
        tuples.append([a, int(b)])
    print(f"Number of clips in {split_type}: {len(tuples)}")
    return tuples


# ----------------------------- dataset -----------------------------


class CHBSeizureDataset(Dataset):
    def __init__(self, input_dir, raw_data_dir, seizure_map, edf_files,
                 time_step_size=1, max_seq_len=12, standardize=False,
                 scaler=None, split="train", data_augment=False,
                 graph_type=None, top_k=3, filter_type="dual_random_walk",
                 sampling_ratio=1, seed=123, use_fft=False, preproc_dir=None,
                 return_raw=False):
        if standardize and scaler is None:
            raise ValueError("standardize=True requires a scaler.")
        if graph_type == "combined":
            raise NotImplementedError(
                "Distance-based 'combined' graph is referential-montage only; "
                "use graph_type='individual' or 'dynamic' for CHB-MIT.")
        if graph_type in ("individual", "dynamic") and top_k is None:
            raise ValueError("top_k required for correlation graphs.")

        self.input_dir = input_dir
        self.raw_data_dir = raw_data_dir
        self.seizure_map = seizure_map
        self.edf_files = edf_files
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.data_augment = data_augment
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir
        self.return_raw = return_raw

        sz_file = FILEMARKER_DIR / f"{split}Set_seq2seq_{max_seq_len}s_sz.txt"
        nosz_file = FILEMARKER_DIR / f"{split}Set_seq2seq_{max_seq_len}s_nosz.txt"
        if not sz_file.exists() or not nosz_file.exists():
            raise FileNotFoundError(
                f"Missing file markers under {FILEMARKER_DIR}. "
                f"Run build_file_markers_chb.py first.")
        self.file_tuples = parseTxtFiles(
            split, str(sz_file), str(nosz_file),
            cv_seed=seed, scale_ratio=sampling_ratio)
        self.size = len(self.file_tuples)

        # Use full bipolar pair name as node id (no '-' split since these ARE
        # bipolar pairs).
        self.sensor_ids = list(INCLUDED_CHANNELS_CHB)
        self._targets = [int(t[1]) for t in self.file_tuples]

    def __len__(self):
        return self.size

    def targets(self):
        return self._targets

    def _random_scale(self, x):
        s = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            x = x + np.log(s)
        else:
            x = x * s
        return x

    def _get_indiv_graph(self, eeg_clip):
        n = len(self.sensor_ids)
        adj = np.eye(n, dtype=np.float32)
        # eeg_clip: (T, N, F) -> (N, T*F)
        flat = np.transpose(eeg_clip, (1, 0, 2)).reshape(n, -1)
        for i in range(n):
            for j in range(i + 1, n):
                xc = comp_xcorr(flat[i], flat[j], mode="valid", normalize=True)
                xc = float(np.asarray(xc).flat[0])  # 'valid' returns 1-element array
                adj[i, j] = xc
                adj[j, i] = xc
        adj = np.abs(adj)
        return keep_topk(adj, top_k=self.top_k, directed=True)

    def _compute_supports(self, adj_mat):
        supports = []
        mats = []
        if self.filter_type == "laplacian":
            mats.append(utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":
            mats.append(utils.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":
            mats.append(utils.calculate_random_walk_matrix(adj_mat).T)
            mats.append(utils.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            mats.append(utils.calculate_scaled_laplacian(adj_mat))
        for m in mats:
            supports.append(torch.FloatTensor(m.toarray()))
        return supports

    def __getitem__(self, idx):
        h5_fn, label = self.file_tuples[idx]

        # parse marker: '<edf_base>.edf_<ci>.h5'
        clip_idx = int(h5_fn.split("_")[-1].split(".h5")[0])
        edf_base = h5_fn.split(".edf")[0]

        # locate raw edf file (strict .edf match)
        edf_matches = [p for p in self.edf_files
                       if os.path.basename(p) == f"{edf_base}.edf"]
        assert len(edf_matches) == 1, (
            f"Expected exactly one edf for {edf_base}, got {edf_matches}")

        raw_clip = None
        if self.preproc_dir is None:
            h5_path = os.path.join(self.input_dir, f"{edf_base}.h5")
            result = _compute_slice(
                h5_path, edf_base, clip_idx, self.seizure_map,
                time_step_size=self.time_step_size,
                clip_len=self.max_seq_len, is_fft=self.use_fft,
                return_raw=self.return_raw)
            if self.return_raw:
                eeg_clip, is_sz, raw_clip = result
            else:
                eeg_clip, is_sz = result
            label = is_sz  # trust the recomputed label
        else:
            with h5py.File(os.path.join(self.preproc_dir, h5_fn), "r") as hf:
                eeg_clip = hf["clip"][()]

        if self.data_augment:
            curr = self._random_scale(eeg_clip.copy())
        else:
            curr = eeg_clip.copy()

        if self.standardize:
            curr = self.scaler.transform(curr)

        x = torch.FloatTensor(curr)
        y = torch.FloatTensor([label])
        seq_len = torch.LongTensor([self.max_seq_len])
        writeout_fn = h5_fn.split(".h5")[0]

        if self.graph_type == "individual":
            adj = self._get_indiv_graph(eeg_clip)
            supports = self._compute_supports(adj)
            T = eeg_clip.shape[0]
            supports_seq = torch.stack(supports).repeat(T, 1, 1, 1)
            adj_seq = np.stack([adj for _ in range(T)])
        elif self.graph_type == "dynamic":
            adj_mats, sups = [], []
            for t in range(eeg_clip.shape[0]):
                adj_t = self._get_indiv_graph(eeg_clip[t][np.newaxis, :])
                sup_t = self._compute_supports(adj_t)
                adj_mats.append(adj_t)
                sups.append(torch.stack(sup_t))
            adj_seq = np.array(adj_mats)
            supports_seq = torch.stack(sups)
        else:
            # No-graph mode (graph_type='none' or any unrecognized value).
            # Return tiny zero placeholders so collation/.to(device) works.
            # Models that opt into this mode must ignore these tensors.
            supports_seq = torch.zeros(1, 1, 1, 1)
            adj_seq = np.zeros((1, 1, 1), dtype=np.float32)

        if self.return_raw:
            raw_t = torch.FloatTensor(raw_clip)
            return (x, y, seq_len, supports_seq, adj_seq, writeout_fn, raw_t)
        return (x, y, seq_len, supports_seq, adj_seq, writeout_fn)


# ----------------------------- entry point -----------------------------


def load_dataset_chb(
        task,
        input_dir,
        raw_data_dir,
        train_batch_size,
        test_batch_size=None,
        time_step_size=1,
        max_seq_len=12,
        standardize=False,
        num_workers=8,
        augmentation=False,
        adj_mat_dir=None,
        graph_type="dynamic",
        top_k=3,
        filter_type="dual_random_walk",
        use_fft=False,
        sampling_ratio=1,
        seed=123,
        preproc_dir=None,
        return_raw=False,
        dense_labels=False):
    """Returns (dataloaders, datasets, scaler) matching the TUSZ interface."""
    if dense_labels:
        raise NotImplementedError(
            "dense_labels=True is not wired up for CHB-MIT; only TUSZ supports "
            "per-second seq2seq labels in this repo.")
    if task != "detection":
        raise NotImplementedError(
            f"CHB-MIT task '{task}' not implemented; only 'detection' is wired up.")

    if standardize:
        # No precomputed means/stds for CHB-MIT — leave hook in case the
        # caller provides them later.
        raise NotImplementedError(
            "standardize=True for CHB-MIT requires precomputed stats; "
            "set standardize=False (which main.py already does).")
    scaler = None

    seizure_map = build_seizure_index(raw_data_dir)
    edf_files = list_edf_files(raw_data_dir)

    dataloaders, datasets = {}, {}
    for split in ("train", "dev", "test"):
        dataset = CHBSeizureDataset(
            input_dir=input_dir,
            raw_data_dir=raw_data_dir,
            seizure_map=seizure_map,
            edf_files=edf_files,
            time_step_size=time_step_size,
            max_seq_len=max_seq_len,
            standardize=False,
            scaler=None,
            split=split,
            data_augment=augmentation if split == "train" else False,
            graph_type=graph_type,
            top_k=top_k,
            filter_type=filter_type,
            sampling_ratio=sampling_ratio,
            seed=seed,
            use_fft=use_fft,
            preproc_dir=preproc_dir,
            return_raw=return_raw,
        )
        shuffle = (split == "train")
        bs = train_batch_size if split == "train" else test_batch_size
        loader = DataLoader(dataset=dataset, shuffle=shuffle,
                            batch_size=bs, num_workers=num_workers)
        dataloaders[split] = loader
        datasets[split] = dataset
    return dataloaders, datasets, scaler
