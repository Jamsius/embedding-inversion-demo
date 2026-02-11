#!/usr/bin/env python3
"""
Interactive Embedding Inversion Demo Server - Multi-Model Support.
Runs on port 8080. Serves a TurboPuffer-style dark UI for real-time diffusion visualization.

Supports both Qwen3-Embedding and EmbeddingGemma models.
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
# Globals - Multi-model support
# ---------------------------------------------------------------------------

app = FastAPI()

# Each model has its own: MODEL, CONFIG, ENCODER_MODEL, ENCODER_TOK, DECODER_TOK
MODELS = {}  # model_key -> dict with model, config, encoder_model, encoder_tok, decoder_tok
DEVICE = None

# Model configurations
MODEL_CONFIGS = {
    "qwen3": {
        "checkpoint_path": str(Path.home() / "checkpoints" / "qwen3_best.pt"),
        "config_path": "configs/v2_qwen3.yaml",
    },
    "gemma": {
        "checkpoint_path": str(Path.home() / "checkpoints" / "gemma_best.pt"),
        "config_path": "configs/v2_gemma.yaml",
    },
}

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


def load_model(model_key):
    """Load a specific model (qwen3 or gemma)."""
    global DEVICE
    
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {DEVICE}")

    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_key.upper()} model")
    print(f"{'='*60}")

    # Load checkpoint
    ckpt_path = cfg["checkpoint_path"]
    print(f"Loading checkpoint {ckpt_path} ...")
    with open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location=DEVICE, pickle_module=pickle)
    config = ckpt["config"]
    print(f"  step={ckpt['step']}, val_loss={ckpt['best_val_loss']:.4f}")
    print(f"  config: vocab={config['model']['vocab_size']}, "
          f"seq_len={config['model']['max_seq_len']}, "
          f"layers={config['model']['num_layers']}, "
          f"embedding_cond_dim={config['model']['embedding_cond_dim']}")

    # Create model
    model = ConditionalMDLM(config).to(DEVICE)
    state = {k: v.float() for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print("  Model loaded OK")
    del ckpt

    # Encoder
    enc_name = config["model"]["encoder_model"]
    print(f"Loading encoder: {enc_name} ...")
    encoder_tok = AutoTokenizer.from_pretrained(enc_name, trust_remote_code=True)
    encoder_model = AutoModel.from_pretrained(enc_name, trust_remote_code=True).to(DEVICE).eval()
    print("  Encoder loaded OK")

    # Decoder tokenizer
    dec_name = config["model"]["decoder_tokenizer"]
    print(f"Loading decoder tokenizer: {dec_name} ...")
    decoder_tok = AutoTokenizer.from_pretrained(dec_name, trust_remote_code=True)
    print("  Decoder tokenizer loaded OK")

    MODELS[model_key] = {
        "model": model,
        "config": config,
        "encoder_model": encoder_model,
        "encoder_tok": encoder_tok,
        "decoder_tok": decoder_tok,
    }
    print(f"{model_key.upper()} ready")


def load_models():
    """Load all models on startup."""
    for model_key in MODEL_CONFIGS:
        load_model(model_key)
    print("\n" + "="*60)
    print("=== Ready ===")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    text: str
    model: str = "qwen3"

class DecodeRequest(BaseModel):
    embedding: List[float]
    steps: int = 32
    model: str = "qwen3"

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
    model_key = req.model.lower()
    if model_key not in MODELS:
        return {"error": f"Unknown model: {model_key}"}
    
    m = MODELS[model_key]
    with torch.no_grad():
        inputs = m["encoder_tok"](
            [req.text], return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(DEVICE)
        out = m["encoder_model"](**inputs)
        emb = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, dim=-1)
    return EncodeResponse(embedding=emb[0].cpu().tolist(), text=req.text)


@app.post("/decode")
async def decode(req: DecodeRequest):
    model_key = req.model.lower()
    if model_key not in MODELS:
        return {"error": f"Unknown model: {model_key}"}
    
    m = MODELS[model_key]
    model = m["model"]
    config = m["config"]
    encoder_model = m["encoder_model"]
    encoder_tok = m["encoder_tok"]
    decoder_tok = m["decoder_tok"]

    async def generate():
        embedding = torch.tensor([req.embedding], device=DEVICE, dtype=torch.float32)
        embedding = F.normalize(embedding, dim=-1)

        L = config["model"]["max_seq_len"]
        mask_id = config["model"]["mask_token_id"]
        steps = max(1, min(req.steps, L))
        per_step = max(1, L // steps)

        ids = torch.full((1, L), mask_id, dtype=torch.long, device=DEVICE)
        unmasked = torch.zeros(L, dtype=torch.bool, device=DEVICE)

        with torch.no_grad():
            for step in range(steps):
                if unmasked.all():
                    break

                logits = model(ids, embedding)
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
                        tok_text = decoder_tok.decode([tid])
                        state = "c" if i in topk_set else "u"  # changed / unchanged
                        tokens.append({"t": tok_text, "s": state})

                # Decode full text (skip mask tokens)
                clean = [t for t in ids[0].cpu().tolist() if t != mask_id]
                text = decoder_tok.decode(clean, skip_special_tokens=True)

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
        final_text = decoder_tok.decode(clean, skip_special_tokens=True)

        with torch.no_grad():
            inputs2 = encoder_tok(
                [final_text], return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(DEVICE)
            out2 = encoder_model(**inputs2)
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
                tokens.append({"t": decoder_tok.decode([tid]), "s": "u"})

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
    return {"status": "ok", "device": str(DEVICE), "models": list(MODELS.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
