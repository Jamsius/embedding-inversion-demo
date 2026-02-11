#!/usr/bin/env python3
"""
Training loop for Conditional MDLM embedding inversion model.

Usage:
    python3 train.py [--config configs/default.yaml] [--resume]
"""

import os
import sys
import time
import argparse
import json
import yaml
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from safetensors.torch import save_file as safetensors_save

from model import ConditionalMDLM, apply_mask
from dataset import create_dataloaders


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr_ratio=0.0):
    """Cosine schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def find_batch_size(config, model, device):
    """Binary search for max batch size that fits in GPU memory."""
    mc = config["model"]
    low, high = 8, 2048
    best = low

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            dummy_ids = torch.randint(0, mc["vocab_size"], (mid, mc["max_seq_len"]), device=device)
            dummy_emb = torch.randn(mid, mc["embedding_cond_dim"], device=device)

            with autocast(dtype=torch.bfloat16):
                hidden = model.forward_hidden(dummy_ids, dummy_emb)
                # Simulate exact chunked CE (same as training loop)
                chunk_size = 256
                h_flat = hidden.view(-1, hidden.shape[-1])
                t_flat = dummy_ids.view(-1)
                total_loss = torch.tensor(0.0, device=device)
                w = model.output_proj.weight
                for ci in range(0, h_flat.shape[0], chunk_size):
                    ce = min(ci + chunk_size, h_flat.shape[0])
                    lc = F.linear(h_flat[ci:ce], w)
                    total_loss = total_loss + F.cross_entropy(lc, t_flat[ci:ce])
                loss = total_loss / (h_flat.shape[0] / chunk_size)
            loss.backward()
            del hidden, h_flat, total_loss, loss
            model.zero_grad()

            best = mid
            print(f"  batch_size={mid} OK", flush=True)
            low = mid + 1
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={mid} OOM", flush=True)
            high = mid - 1
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  batch_size={mid} error: {e}", flush=True)
            high = mid - 1

    safe = max(32, int(best * 0.85))
    print(f"Max batch size: {best}, using: {safe}", flush=True)
    return safe


def _meta(step, best_val_loss, config):
    """Build metadata dict for safetensors (all values must be strings)."""
    mc = config.get("model", {})
    return {
        "step": str(step),
        "best_val_loss": f"{best_val_loss:.6f}",
        "encoder_model": str(mc.get("encoder_model", "unknown")),
        "decoder_tokenizer": str(mc.get("decoder_tokenizer", "unknown")),
        "vocab_size": str(mc.get("vocab_size", 0)),
        "hidden_dim": str(mc.get("hidden_dim", 0)),
        "num_layers": str(mc.get("num_layers", 0)),
        "max_seq_len": str(mc.get("max_seq_len", 0)),
        "embedding_cond_dim": str(mc.get("embedding_cond_dim", 0)),
        "config_json": json.dumps(config, default=str),
    }


def save_checkpoint(path, step, best_val_loss, model, ema_model, optimizer, scaler, config):
    """Save full checkpoint for resuming training (.pt, includes optimizer state)."""
    torch.save({
        "step": step,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }, path)


def save_ema(path, step, best_val_loss, ema_model, config):
    """Save inference-only EMA weights as safetensors with metadata."""
    st_path = path.replace(".pt", ".safetensors")
    safetensors_save(ema_model.state_dict(), st_path, metadata=_meta(step, best_val_loss, config))


def train(config, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    mc = config["model"]
    tc = config["training"]

    # Build model
    model = ConditionalMDLM(config).to(device)
    total_params, trainable_params = model.count_params()
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable", flush=True)

    # EMA model
    ema_decay = tc.get("ema_decay", 0.9999)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print(f"EMA decay: {ema_decay}", flush=True)

    # Batch size: use config value directly (auto-tune unreliable with chunked CE)
    batch_size = tc.get("batch_size", 128)
    tc["batch_size"] = batch_size
    print(f"Using batch size: {batch_size}", flush=True)

    # Data
    print("Loading data...", flush=True)
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches", flush=True)
    print(f"Val: {len(val_loader.dataset):,} samples", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    scaler = GradScaler()

    # Gradient accumulation
    grad_accum = tc.get("grad_accum", 1)
    effective_batch = batch_size * grad_accum
    print(f"Effective batch size: {effective_batch} (micro={batch_size} x accum={grad_accum})", flush=True)

    # Resume
    start_step = 0
    ckpt_dir = config.get("_ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}", flush=True)

    if resume:
        ckpt_path = f"{ckpt_dir}/latest.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"{ckpt_dir}/best.pt"  # fallback
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}...", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model"]
            # Strip _orig_mod prefix if present (from compiled checkpoints)
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_sd)
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt["step"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if "ema_model" in ckpt:
                ema_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema_model"].items()}
                ema_model.load_state_dict(ema_sd)
                print("Loaded EMA weights", flush=True)
            print(f"Resumed at step {start_step}", flush=True)

    # Training loop
    model.train()
    step = start_step
    mask_token_id = mc["mask_token_id"]
    max_steps = tc["max_steps"]
    log_every = tc["log_every"]

    if start_step == 0:
        best_val_loss = float("inf")
    eval_every = tc.get("eval_every", 500)
    early_stop_patience = tc.get("early_stop_patience", 5000)
    best_step = start_step  # step when best_val_loss was last improved
    running_loss = 0.0
    running_acc = 0.0
    running_count = 0
    micro_step = 0
    t0_global = time.time()
    total_samples = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    epoch = 0

    print(f"\n=== Training started (step {step}/{max_steps}) ===", flush=True)

    while step < max_steps:
        # Get batch (restart iterator if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            print(f"--- Epoch {epoch} complete (step {step}) ---", flush=True)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        token_ids = batch["token_ids"].to(device)
        embedding = batch["embedding"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        # Apply random masking
        masked_ids, target_mask, mask_ratio = apply_mask(token_ids, mask_token_id, padding_mask)

        # Forward
        if micro_step == 0:
            min_lr_ratio = tc.get("min_lr_ratio", 0.0)
            lr = get_lr(step, tc["warmup_steps"], max_steps, tc["lr"], min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            # Get hidden states instead of full logits to save memory
            hidden = model.forward_hidden(masked_ids, embedding, padding_mask)

            # Chunked cross-entropy: compute loss without materializing full [B*32, 250K] logits
            chunk_size = 256  # positions per chunk
            total_positions = hidden.shape[0] * hidden.shape[1]  # B * seq_len
            hidden_flat = hidden.view(-1, hidden.shape[-1])  # [B*32, hidden]
            targets_flat = token_ids.view(-1)  # [B*32]
            mask_flat = target_mask.view(-1).float()  # [B*32]

            total_loss = torch.tensor(0.0, device=device)
            total_correct = 0
            total_masked = mask_flat.sum().item()

            for i in range(0, total_positions, chunk_size):
                end = min(i + chunk_size, total_positions)
                h_chunk = hidden_flat[i:end]  # [chunk, hidden]
                t_chunk = targets_flat[i:end]  # [chunk]
                m_chunk = mask_flat[i:end]  # [chunk]

                # Compute logits only for this chunk (memory efficient)
                w = model.output_proj.weight  # [vocab, hidden] - tied with token_embed
                logits_chunk = F.linear(h_chunk, w)  # [chunk, vocab]
                loss_chunk = F.cross_entropy(logits_chunk, t_chunk, reduction="none")
                total_loss = total_loss + (loss_chunk * m_chunk).sum()

                with torch.no_grad():
                    preds_chunk = logits_chunk.argmax(-1)
                    total_correct += ((preds_chunk == t_chunk) * m_chunk.bool()).sum().item()

            loss = total_loss / max(total_masked, 1)
            # MDLM 1/t loss weighting (Rao-Blackwellized ELBO)
            loss_weight = (1.0 / mask_ratio.squeeze(1)).mean()
            loss = loss * loss_weight
            loss = loss / grad_accum  # scale for accumulation

        scaler.scale(loss).backward()

        running_loss += loss.item() * grad_accum
        running_acc += total_correct / max(total_masked, 1)
        running_count += 1
        total_samples += token_ids.shape[0]
        micro_step += 1

        if micro_step < grad_accum:
            continue

        # Optimizer step after accumulation
        micro_step = 0
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        # EMA update
        with torch.no_grad():
            for ep, mp in zip(ema_model.parameters(), model.parameters()):
                ep.mul_(ema_decay).add_(mp, alpha=1 - ema_decay)
        step += 1

        # Log
        if step % log_every == 0:
            avg_loss = running_loss / running_count
            avg_acc = running_acc / running_count
            global_elapsed = time.time() - t0_global
            rate = total_samples / global_elapsed
            print(
                f"step {step}/{max_steps} | loss {avg_loss:.4f} | acc {avg_acc:.3f} | "
                f"lr {lr:.2e} | {rate:.0f} samples/sec | "
                f"elapsed {global_elapsed/60:.1f}min",
                flush=True
            )
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0

        # Validation & save best
        if step % eval_every == 0:
            ema_model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for i, vb in enumerate(val_loader):
                    if i >= 20:  # 20 batches for speed
                        break
                    vids = vb["token_ids"].to(device)
                    vemb = vb["embedding"].to(device)
                    vpad = vb["padding_mask"].to(device)
                    vm_ids, vm_mask, _ = apply_mask(vids, mask_token_id, vpad)

                    with autocast(dtype=torch.bfloat16):
                        vhidden = ema_model.forward_hidden(vm_ids, vemb, vpad)
                        # Chunked CE for validation (avoid OOM)
                        vh_flat = vhidden.view(-1, vhidden.shape[-1])
                        vt_flat = vids.view(-1)
                        vm_flat = vm_mask.view(-1).float()
                        vw = ema_model.output_proj.weight
                        vtotal = torch.tensor(0.0, device=device)
                        for vi in range(0, vh_flat.shape[0], 256):
                            ve = min(vi + 256, vh_flat.shape[0])
                            vlc = F.linear(vh_flat[vi:ve], vw)
                            vtotal = vtotal + (F.cross_entropy(vlc, vt_flat[vi:ve], reduction="none") * vm_flat[vi:ve]).sum()
                        vloss = vtotal / vm_flat.sum().clamp(min=1)

                    val_loss += vloss.item()
                    val_count += 1

            avg_val = val_loss / max(val_count, 1)
            improved = " [BEST]" if avg_val < best_val_loss else ""
            print(f"  val_loss: {avg_val:.4f}{improved}", flush=True)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_checkpoint(f"{ckpt_dir}/best.pt", step, best_val_loss, model, ema_model, optimizer, scaler, config)
                save_ema(f"{ckpt_dir}/best_ema.pt", step, best_val_loss, ema_model, config)
                print(f"  Saved best.pt + best_ema.pt (step {step}, val_loss {avg_val:.4f})", flush=True)
                best_step = step

            # Early stopping check
            if step - best_step >= early_stop_patience and step > early_stop_patience:
                print(f"\n=== Early stopping: no improvement for {step - best_step} steps (patience={early_stop_patience}) ===", flush=True)
                save_checkpoint(f"{ckpt_dir}/final.pt", step, best_val_loss, model, ema_model, optimizer, scaler, config)
                save_ema(f"{ckpt_dir}/final_ema.pt", step, best_val_loss, ema_model, config)
                print(f"Saved final.pt + final_ema.pt (best val_loss: {best_val_loss:.4f} at step {best_step})", flush=True)
                return

            # Save latest for resume
            save_checkpoint(f"{ckpt_dir}/latest.pt", step, best_val_loss, model, ema_model, optimizer, scaler, config)

            # Save milestone checkpoints for paper experiments
            milestones = {10000, 25000, 50000, 100000, 200000}
            if step in milestones:
                mpath = f"{ckpt_dir}/step_{step}.pt"
                save_checkpoint(mpath, step, best_val_loss, model, ema_model, optimizer, scaler, config)
                save_ema(f"{ckpt_dir}/step_{step}_ema.pt", step, best_val_loss, ema_model, config)
                print(f"  Saved milestone: {mpath} + ema", flush=True)

            model.train()

    print(f"\n=== Training complete ({step} steps) ===", flush=True)

    # Save final
    save_checkpoint(f"{ckpt_dir}/final.pt", step, best_val_loss, model, ema_model, optimizer, scaler, config)
    save_ema(f"{ckpt_dir}/final_ema.pt", step, best_val_loss, ema_model, config)
    print(f"Saved final.pt + final_ema.pt (best val_loss: {best_val_loss:.4f})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Derive checkpoint dir from config filename (e.g. v2_gemma.yaml -> checkpoints_v2_gemma)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    config["_ckpt_dir"] = f"checkpoints_{config_name}"

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
