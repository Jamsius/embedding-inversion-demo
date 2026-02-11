"""
Dataset for embedding inversion training.
Reads pre-converted numpy .npy files for instant loading.
"""

import os
import glob
import bisect
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingInversionDataset(Dataset):
    """
    Memory-mapped numpy dataset. Loading is instant.
    """

    def __init__(self, data_dir, max_seq_len=32, val=False, val_split=0.01,
                 pad_token_id=1, bos_token_id=0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Convention: npy files are in {data_dir}_npy/
        if os.path.isdir(data_dir + "_npy"):
            npy_dir = data_dir + "_npy"
        elif os.path.isdir(data_dir):
            npy_dir = data_dir
        else:
            npy_dir = data_dir.rstrip("/") + "_npy"

        # Find all npy chunk pairs
        tid_files = sorted(glob.glob(os.path.join(npy_dir, "token_ids_*.npy")))
        emb_files = sorted(glob.glob(os.path.join(npy_dir, "embeddings_*.npy")))
        assert len(tid_files) == len(emb_files), f"Mismatch: {len(tid_files)} vs {len(emb_files)}"
        assert len(tid_files) > 0, f"No npy files found in {npy_dir}"

        # Memory-map all chunks (no RAM cost until accessed)
        self.tid_maps = [np.load(f, mmap_mode='r') for f in tid_files]
        self.emb_maps = [np.load(f, mmap_mode='r') for f in emb_files]

        # Build index
        self.chunk_sizes = [m.shape[0] for m in self.tid_maps]
        self.chunk_offsets = []
        offset = 0
        for s in self.chunk_sizes:
            self.chunk_offsets.append(offset)
            offset += s
        self.total_rows = offset

        # Train/val split
        val_count = int(self.total_rows * val_split)
        if val:
            self.start_idx = self.total_rows - val_count
            self.length = val_count
        else:
            self.start_idx = 0
            self.length = self.total_rows - val_count

    def __len__(self):
        return self.length

    def _find_chunk(self, global_idx):
        chunk_idx = bisect.bisect_right(self.chunk_offsets, global_idx) - 1
        if chunk_idx < 0:
            raise IndexError(f"Index {global_idx} out of range")
        return chunk_idx, global_idx - self.chunk_offsets[chunk_idx]

    def __getitem__(self, idx):
        global_idx = self.start_idx + idx
        chunk_idx, local_idx = self._find_chunk(global_idx)

        token_ids = torch.from_numpy(self.tid_maps[chunk_idx][local_idx].copy()).long()
        embedding = torch.from_numpy(self.emb_maps[chunk_idx][local_idx].copy()).float()

        # Mark padding positions (model-specific pad/bos tokens)
        pad_id = self.pad_token_id
        padding_mask = (token_ids == pad_id)
        if self.bos_token_id is not None:
            padding_mask = padding_mask | (token_ids == self.bos_token_id)

        return {
            "token_ids": token_ids,
            "embedding": embedding,
            "padding_mask": padding_mask,
        }


def create_dataloaders(config):
    dc = config["data"]
    tc = config["training"]
    mc = config["model"]

    # Read pad/bos from meta.json if available, else use defaults
    npy_dir = dc["data_dir"] + "_npy" if os.path.isdir(dc["data_dir"] + "_npy") else dc["data_dir"]
    meta_path = os.path.join(npy_dir, "meta.json")
    pad_id, bos_id = 1, 0  # XLM-R defaults
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pad_id = meta.get("pad_id", 1)
        bos_id = None  # only XLM-R uses BOS=0

    train_ds = EmbeddingInversionDataset(
        dc["data_dir"], mc["max_seq_len"],
        val=False, val_split=dc["val_split"],
        pad_token_id=pad_id, bos_token_id=bos_id
    )
    val_ds = EmbeddingInversionDataset(
        dc["data_dir"], mc["max_seq_len"],
        val=True, val_split=dc["val_split"],
        pad_token_id=pad_id, bos_token_id=bos_id
    )

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc["num_workers"], pin_memory=True
    )

    return train_loader, val_loader
