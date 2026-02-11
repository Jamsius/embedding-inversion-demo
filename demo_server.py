#!/usr/bin/env python3
"""
Interactive Embedding Inversion Demo Server.
Runs on port 8080. Serves a TurboPuffer-style dark UI for real-time diffusion visualization.

Architecture matching the qwen3_best.pt checkpoint:
  - Standalone DiT-style transformer (no mmBERT dependency)
  - AdaLN conditioning (scale + shift, no alpha gate)
  - 8 blocks, hidden_dim=768, num_heads=12, ff_dim=3072
  - vocab_size=151936, max_seq_len=32
"""

import sys
import os
import pickle
import yaml
import json
import math
import random
import asyncio
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# ---------------------------------------------------------------------------
# Model architecture (matches checkpoint state_dict exactly)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive LayerNorm: norm(x) * (1+scale) + shift, conditioned on cond."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, cond):
        # cond: [B, cond_dim] -> [B, 2*hidden]
        params = self.proj(cond).unsqueeze(1)  # [B, 1, 2*hidden]
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class Block(nn.Module):
    """Transformer block with AdaLN conditioning."""
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x, cond):
        # Self-attention with AdaLN
        normed = self.adaln1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        # Feed-forward with AdaLN
        normed = self.adaln2(x, cond)
        x = x + self.ff(normed)
        return x


class ConditionalMDLM(nn.Module):
    """Conditional Masked Diffusion Language Model - matches checkpoint exactly."""
    def __init__(self, config):
        super().__init__()
        mc = config["model"]
        self.vocab_size = mc["vocab_size"]
        self.hidden_dim = mc["hidden_dim"]
        self.max_seq_len = mc["max_seq_len"]
        self.mask_token_id = mc["mask_token_id"]

        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(mc["embedding_cond_dim"], self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            Block(self.hidden_dim, mc["num_heads"], mc["ff_dim"], mc.get("dropout", 0.0))
            for _ in range(mc["num_layers"])
        ])
        self.final_norm = AdaLN(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def forward(self, input_ids, cond_embedding, padding_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        cond = self.cond_proj(cond_embedding)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x, cond)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Encoder helper
# ---------------------------------------------------------------------------

def last_token_pool(hidden, attention_mask):
    """Pool using last non-padding token (Qwen3-Embedding style)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens]


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = FastAPI()

MODEL = None
CONFIG = None
ENCODER_MODEL = None
ENCODER_TOK = None
DECODER_TOK = None
DEVICE = None

CHECKPOINT_PATH = str(Path.home() / "checkpoints" / "qwen3_best.pt")

SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming the world",
    "Machine learning models can understand language",
    "I love spending time in nature during autumn",
    "Coffee is the fuel that powers innovation",
    "Music transcends all cultural boundaries",
    "The ocean waves crash against the rocky shore",
    "Reading opens doors to infinite worlds",
    "Technology connects people across vast distances",
    "Dreams are the seeds of future realities",
    "Stars shine brightest in the darkest nights",
    "Knowledge is the most powerful currency",
    "Every great journey begins with a single step",
    "Creativity flourishes when constraints are removed",
    "The universe is full of unsolved mysteries",
    "Quantum computers will revolutionize cryptography",
    "Fresh bread from the oven smells wonderful",
    "Photography captures moments that last forever",
    "Mathematics is the language of the universe",
    "Kindness costs nothing but means everything",
]


def load_models():
    global MODEL, CONFIG, ENCODER_MODEL, ENCODER_TOK, DECODER_TOK, DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Load checkpoint (needs pickle_module workaround for this file)
    print(f"Loading checkpoint {CHECKPOINT_PATH} ...")
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = torch.load(f, map_location=DEVICE, pickle_module=pickle)
    CONFIG = ckpt["config"]
    print(f"  step={ckpt['step']}, val_loss={ckpt['best_val_loss']:.4f}")
    print(f"  config: vocab={CONFIG['model']['vocab_size']}, "
          f"seq_len={CONFIG['model']['max_seq_len']}, "
          f"layers={CONFIG['model']['num_layers']}")

    MODEL = ConditionalMDLM(CONFIG).to(DEVICE)
    state = {k: v.float() for k, v in ckpt["ema_state_dict"].items()}
    MODEL.load_state_dict(state, strict=True)
    MODEL.eval()
    print("  Model loaded OK")
    del ckpt

    # Encoder
    enc_name = CONFIG["model"]["encoder_model"]
    print(f"Loading encoder: {enc_name} ...")
    ENCODER_TOK = AutoTokenizer.from_pretrained(enc_name, trust_remote_code=True)
    ENCODER_MODEL = AutoModel.from_pretrained(enc_name, trust_remote_code=True).to(DEVICE).eval()
    print("  Encoder loaded OK")

    # Decoder tokenizer (same model name, just the tokenizer)
    dec_name = CONFIG["model"]["decoder_tokenizer"]
    print(f"Loading decoder tokenizer: {dec_name} ...")
    DECODER_TOK = AutoTokenizer.from_pretrained(dec_name, trust_remote_code=True)
    print("  Decoder tokenizer loaded OK")

    print("=== Ready ===")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    text: str

class DecodeRequest(BaseModel):
    embedding: List[float]
    steps: int = 32

class EncodeResponse(BaseModel):
    embedding: List[float]
    text: str


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest):
    with torch.no_grad():
        inputs = ENCODER_TOK(
            [req.text], return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(DEVICE)
        out = ENCODER_MODEL(**inputs)
        emb = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, dim=-1)
    return EncodeResponse(embedding=emb[0].cpu().tolist(), text=req.text)


@app.post("/decode")
async def decode(req: DecodeRequest):
    async def generate():
        embedding = torch.tensor([req.embedding], device=DEVICE, dtype=torch.float32)
        embedding = F.normalize(embedding, dim=-1)

        L = CONFIG["model"]["max_seq_len"]
        mask_id = CONFIG["model"]["mask_token_id"]
        steps = max(1, min(req.steps, L))
        per_step = max(1, L // steps)

        ids = torch.full((1, L), mask_id, dtype=torch.long, device=DEVICE)
        unmasked = torch.zeros(L, dtype=torch.bool, device=DEVICE)

        with torch.no_grad():
            for step in range(steps):
                if unmasked.all():
                    break

                logits = MODEL(ids, embedding)
                probs = F.softmax(logits[0], dim=-1)
                confidence, preds = probs.max(dim=-1)
                confidence[unmasked] = -1.0

                k = min(per_step, (~unmasked).sum().item())
                if k == 0:
                    break
                _, topk = confidence.topk(k)
                topk_set = set(topk.cpu().tolist())

                ids[0, topk] = preds[topk]
                unmasked[topk] = True

                # Build per-token info
                tokens = []
                for i in range(L):
                    tid = ids[0, i].item()
                    if tid == mask_id:
                        tokens.append({"t": "[MASK]", "s": "m"})  # masked
                    else:
                        tok_text = DECODER_TOK.decode([tid])
                        state = "c" if i in topk_set else "u"  # changed / unchanged
                        tokens.append({"t": tok_text, "s": state})

                # Decode full text (skip mask tokens)
                clean = [t for t in ids[0].cpu().tolist() if t != mask_id]
                text = DECODER_TOK.decode(clean, skip_special_tokens=True)

                evt = {
                    "step": step,
                    "total": steps,
                    "tokens": tokens,
                    "text": text,
                    "progress": float(unmasked.sum().item()) / L,
                }
                yield f"data: {json.dumps(evt)}\n\n"
                await asyncio.sleep(0.08)

        # Final: compute cosine similarity by re-encoding decoded text
        clean = [t for t in ids[0].cpu().tolist() if t != mask_id]
        final_text = DECODER_TOK.decode(clean, skip_special_tokens=True)

        with torch.no_grad():
            inputs2 = ENCODER_TOK(
                [final_text], return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(DEVICE)
            out2 = ENCODER_MODEL(**inputs2)
            emb2 = last_token_pool(out2.last_hidden_state, inputs2["attention_mask"])
            emb2 = F.normalize(emb2, dim=-1)
            cos_sim = F.cosine_similarity(embedding, emb2).item()

        # Build final token list
        tokens = []
        for i in range(L):
            tid = ids[0, i].item()
            if tid == mask_id:
                tokens.append({"t": "[MASK]", "s": "m"})
            else:
                tokens.append({"t": DECODER_TOK.decode([tid]), "s": "u"})

        evt = {
            "step": steps,
            "total": steps,
            "tokens": tokens,
            "text": final_text,
            "progress": 1.0,
            "cosine_similarity": round(cos_sim, 4),
            "done": True,
        }
        yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/random")
async def get_random():
    return {"text": random.choice(SAMPLE_SENTENCES)}


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
