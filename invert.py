#!/usr/bin/env python3
"""
Minimal embedding inversion: embedding vector -> text.
Only depends on: torch, transformers, yaml.

Usage:
    python invert.py "any text to test round-trip"
    python invert.py --embedding path/to/embedding.npy
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from model import ConditionalMDLM

CHECKPOINT = "checkpoints/v1_inference_fp16.pt"


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def invert(embedding, model, config, steps=50):
    """Invert a [1, 1024] embedding back to token ids."""
    device = embedding.device
    L = config["model"]["max_seq_len"]
    mask_id = config["model"]["mask_token_id"]

    ids = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    unmasked = torch.zeros(L, dtype=torch.bool, device=device)
    per_step = max(1, L // steps)

    for step in range(steps):
        if unmasked.all():
            break
        logits = model(ids, embedding)
        probs = F.softmax(logits[0], dim=-1)
        confidence, preds = probs.max(dim=-1)
        confidence[unmasked] = -1
        k = min(per_step, (~unmasked).sum().item())
        _, topk = confidence.topk(k)
        ids[0, topk] = preds[topk]
        unmasked[topk] = True

    return ids[0]


def main():
    parser = argparse.ArgumentParser(description="Invert embeddings to text")
    parser.add_argument("text", nargs="?", help="Text to embed then invert (round-trip test)")
    parser.add_argument("--embedding", help="Path to .npy embedding file")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    if not args.text and not args.embedding:
        parser.error("Provide text or --embedding")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safetensors_load
        import json as _json
        tensors = safetensors_load(args.checkpoint, device=str(device))
        # Read metadata from safetensors header
        from safetensors import safe_open
        with safe_open(args.checkpoint, framework="pt") as f:
            meta = f.metadata()
        config = _json.loads(meta["config_json"])
        model = ConditionalMDLM(config).to(device).eval()
        state = {k: v.float() for k, v in tensors.items()}
        model.load_state_dict(state)
        print(f"Loaded safetensors (step {meta['step']}, val_loss {meta['best_val_loss']})")
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = ConditionalMDLM(config).to(device).eval()
        state = ckpt.get("ema_model", ckpt.get("model"))
        state = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint (step {ckpt['step']}, val_loss {ckpt['best_val_loss']:.4f})")

    # Get embedding
    if args.text:
        jina_tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3",
                                                  trust_remote_code=True)
        jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3",
                                                trust_remote_code=True).to(device).eval()
        inputs = jina_tok([args.text], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        out = jina_model(**inputs)
        emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, dim=-1)
        print(f"Input: {args.text}")
    else:
        import numpy as np
        emb = torch.from_numpy(np.load(args.embedding)).to(device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb = F.normalize(emb, dim=-1)

    # Invert
    pred_ids = invert(emb, model, config, steps=args.steps)
    xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    clean = [t for t in pred_ids.cpu().tolist() if t not in (0, 1, config["model"]["mask_token_id"])]
    text = xlmr_tok.decode(clean, skip_special_tokens=True)
    print(f"Output: {text}")

    # Cosine similarity if round-trip
    if args.text:
        inputs2 = jina_tok([text], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        out2 = jina_model(**inputs2)
        emb2 = mean_pool(out2.last_hidden_state, inputs2["attention_mask"])
        emb2 = F.normalize(emb2, dim=-1)
        cos = F.cosine_similarity(emb, emb2).item()
        print(f"Cosine similarity: {cos:.4f}")


if __name__ == "__main__":
    main()
