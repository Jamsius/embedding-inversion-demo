#!/usr/bin/env python3
"""
Two-stage multilingual data preparation for embedding inversion.

Stage 1: Download raw texts from mc4/c4 (CPU, no GPU needed)
  - Downloads N samples across K languages
  - Saves raw texts as JSON chunks
  - No tokenization, no encoder dependency

Stage 2: Tokenize + Encode for a specific encoder (GPU)
  - Reads raw texts from Stage 1
  - Tokenizes with encoder's tokenizer, truncates to seq_len
  - Decodes back to get aligned text (encoder sees exactly what decoder targets)
  - Encodes aligned text into embeddings
  - Saves token_ids + embeddings as npy files

Usage:
    # Stage 1: download multilingual texts (run once)
    python3 prepare_data_fast.py --stage 1 --langs en,zh,de,fr,ja,es,ru,ko,ar,pt --n-samples 2000000

    # Stage 2: encode for a specific model (run per encoder)
    python3 prepare_data_fast.py --stage 2 --config configs/v2_jina.yaml --encode-batch 1024

    # Both stages at once
    python3 prepare_data_fast.py --config configs/v2_jina.yaml --langs en --n-samples 2000000
"""

import os
import json
import argparse
import time
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CHUNK_SIZE = 200000
DEFAULT_LANGS = ["en", "zh", "de", "fr", "ja", "es", "ru", "ko", "ar", "pt"]
DEFAULT_RAW_DIR = "data_raw"


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def last_token_pool(hidden, mask):
    seq_len = mask.sum(dim=1, keepdim=True)
    indices = (seq_len - 1).long().unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, indices).squeeze(1)


def detect_pool_method(model_name):
    if "qwen" in model_name.lower():
        return "last"
    return "mean"


def stage1_download(args):
    """Download raw texts from mc4/c4. No tokenization."""
    from datasets import load_dataset

    raw_dir = args.raw_dir
    os.makedirs(raw_dir, exist_ok=True)

    langs = args.langs.split(",")
    n_total = args.n_samples
    n_per_lang = n_total // len(langs)
    remainder = n_total - n_per_lang * len(langs)

    print(f"=== Stage 1: Download Raw Texts ===", flush=True)
    print(f"Languages: {langs}", flush=True)
    print(f"Total samples: {n_total:,} ({n_per_lang:,} per lang, +{remainder} for first)", flush=True)
    print(f"Output: {raw_dir}/", flush=True)

    all_texts = []
    t0 = time.time()

    for i, lang in enumerate(langs):
        target = n_per_lang + (remainder if i == 0 else 0)
        print(f"\n[{lang}] Downloading {target:,} samples...", flush=True)

        # Try mc4 first, fall back to c4
        try:
            ds = load_dataset("allenai/mc4", lang, split="train", streaming=True)
            source = "mc4"
        except Exception:
            try:
                ds = load_dataset("allenai/c4", lang, split="train", streaming=True)
                source = "c4"
            except Exception as e:
                print(f"  SKIP {lang}: {e}", flush=True)
                continue

        print(f"  Source: {source}", flush=True)
        collected = 0
        skipped = 0

        for example in ds:
            if collected >= target:
                break

            text = example["text"].strip()
            # Rough filter: keep texts between 20-500 chars
            # (generous range, exact truncation happens in Stage 2)
            if len(text) < 20 or len(text) > 500:
                skipped += 1
                continue

            all_texts.append({"text": text, "lang": lang})
            collected += 1

            if collected % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  {collected:,}/{target:,} | {elapsed/60:.1f}min", flush=True)

        elapsed = time.time() - t0
        print(f"  [{lang}] Done: {collected:,} collected, {skipped:,} skipped | {elapsed/60:.1f}min", flush=True)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_texts)

    # Save in chunks
    chunk_id = 0
    for start in range(0, len(all_texts), CHUNK_SIZE):
        chunk = all_texts[start:start + CHUNK_SIZE]
        path = os.path.join(raw_dir, f"texts_{chunk_id:04d}.json")
        with open(path, "w") as f:
            json.dump(chunk, f, ensure_ascii=False)
        print(f"Saved chunk {chunk_id}: {len(chunk):,} samples", flush=True)
        chunk_id += 1

    # Metadata
    meta = {
        "n_chunks": chunk_id,
        "total_samples": len(all_texts),
        "langs": langs,
        "n_per_lang": n_per_lang,
        "source": "mc4/c4",
    }
    with open(os.path.join(raw_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nStage 1 done! {len(all_texts):,} samples in {chunk_id} chunks, "
          f"{elapsed/60:.1f}min", flush=True)
    return raw_dir, chunk_id


def stage2_encode(config, args):
    """Tokenize + encode raw texts for a specific encoder."""
    mc = config["model"]
    seq_len = mc["max_seq_len"]
    encoder_name = mc["encoder_model"]
    decoder_tok_name = mc.get("decoder_tokenizer", encoder_name)
    output_dir = config["data"]["data_dir"]
    raw_dir = args.raw_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    with open(os.path.join(raw_dir, "meta.json")) as f:
        raw_meta = json.load(f)
    n_chunks = raw_meta["n_chunks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Stage 2: Tokenize + Encode ({device}) ===", flush=True)
    print(f"Encoder: {encoder_name}", flush=True)
    print(f"Decoder tokenizer: {decoder_tok_name}", flush=True)
    print(f"Seq len: {seq_len}", flush=True)
    print(f"Raw data: {raw_dir}/ ({n_chunks} chunks)", flush=True)
    print(f"Output: {output_dir}/", flush=True)

    # Load tokenizer and model
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tok_name, trust_remote_code=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
    encoder_model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True).to(device).eval()

    pad_id = decoder_tokenizer.pad_token_id
    eos_id = decoder_tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id
    vocab_size = len(decoder_tokenizer)

    pool_method = detect_pool_method(encoder_name)
    print(f"Pool method: {pool_method}", flush=True)
    print(f"Vocab: {vocab_size:,}, pad={pad_id}, eos={eos_id}", flush=True)

    # Verify embedding dim
    test_inputs = encoder_tokenizer(["test"], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        test_out = encoder_model(**test_inputs)
        if pool_method == "last":
            test_emb = last_token_pool(test_out.last_hidden_state, test_inputs["attention_mask"])
        else:
            test_emb = mean_pool(test_out.last_hidden_state, test_inputs["attention_mask"])
    emb_dim = test_emb.shape[-1]
    print(f"Embedding dim: {emb_dim}", flush=True)

    encode_batch = args.encode_batch
    t0 = time.time()
    total_processed = 0
    total_skipped = 0
    out_chunk_id = 0

    for chunk_id in range(n_chunks):
        # Load raw texts
        with open(os.path.join(raw_dir, f"texts_{chunk_id:04d}.json")) as f:
            items = json.load(f)

        # Tokenize, truncate, align
        aligned_texts = []
        token_ids_list = []

        for item in items:
            text = item["text"] if isinstance(item, dict) else item

            # Tokenize with decoder tokenizer, truncate to seq_len - 1 (leave room for EOS)
            raw_ids = decoder_tokenizer.encode(text, add_special_tokens=False)
            max_content = seq_len - 1
            raw_ids = raw_ids[:max_content]
            if len(raw_ids) < 5:
                total_skipped += 1
                continue

            # Aligned text: decode truncated IDs back
            aligned_text = decoder_tokenizer.decode(raw_ids)

            # Target token IDs: content + EOS + padding
            target_ids = list(raw_ids)
            target_ids.append(eos_id)
            target_ids += [pad_id] * (seq_len - len(target_ids))
            token_ids = np.array(target_ids[:seq_len], dtype=np.int32)

            aligned_texts.append(aligned_text)
            token_ids_list.append(token_ids)

        if not aligned_texts:
            continue

        token_ids_array = np.array(token_ids_list, dtype=np.int32)

        # Encode in batches
        all_embs = []
        for i in range(0, len(aligned_texts), encode_batch):
            batch_texts = aligned_texts[i:i + encode_batch]
            inputs = encoder_tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = encoder_model(**inputs)
                if pool_method == "last":
                    emb = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
                else:
                    emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy().astype(np.float32))

        emb_array = np.concatenate(all_embs, axis=0)

        # Save
        np.save(os.path.join(output_dir, f"token_ids_{out_chunk_id:04d}.npy"), token_ids_array)
        np.save(os.path.join(output_dir, f"embeddings_{out_chunk_id:04d}.npy"), emb_array)

        total_processed += len(aligned_texts)
        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  Chunk {out_chunk_id}: {len(aligned_texts):,} samples | "
              f"{total_processed:,} total | "
              f"{rate:.0f} samples/sec | {elapsed/60:.1f}min", flush=True)
        out_chunk_id += 1

    # Save metadata
    meta = {
        "n_chunks": out_chunk_id,
        "total_samples": total_processed,
        "skipped": total_skipped,
        "seq_len": seq_len,
        "encoder_model": encoder_name,
        "decoder_tokenizer": decoder_tok_name,
        "embedding_dim": int(emb_dim),
        "vocab_size": vocab_size,
        "pad_id": int(pad_id),
        "eos_id": int(eos_id),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nStage 2 done! {total_processed:,} samples ({total_skipped:,} skipped), "
          f"{elapsed/60:.1f}min ({total_processed/elapsed:.0f} samples/sec)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Model config yaml (required for stage 2)")
    parser.add_argument("--stage", type=int, default=0, help="0=both, 1=download only, 2=encode only")
    parser.add_argument("--n-samples", type=int, default=2000000)
    parser.add_argument("--langs", default="en", help="Comma-separated language codes")
    parser.add_argument("--encode-batch", type=int, default=1024)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Directory for raw texts")
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if args.stage in (0, 1):
        stage1_download(args)

    if args.stage in (0, 2):
        if config is None:
            print("ERROR: --config required for stage 2", flush=True)
            return
        stage2_encode(config, args)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
