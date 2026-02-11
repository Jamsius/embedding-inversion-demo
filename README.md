# Embedding Inversion via Conditional Masked Diffusion

Embeddings are widely assumed to be safe, irreversible representations. This demo shows they can be inverted to recover the original text.

## Motivation

Production systems routinely treat embeddings as anonymized: vector databases transmit them across organizational boundaries, API services cache them without encryption, distributed search systems share them with third-party providers. These practices assume embeddings are irreversible.

They are not. This implementation recovers 69.7% of tokens and achieves 0.83 cosine similarity from a single 1024-dimensional vector, without any access to the target encoder at inference time.

## Method

We frame embedding inversion as conditional masked diffusion. Starting from a fully masked sequence, a denoising model iteratively reveals tokens at all positions in parallel, conditioned on the target embedding vector via adaptive layer normalization.

The key difference from prior work: correction is built into the diffusion process itself. Each denoising step refines all positions simultaneously using global context, without ever re-embedding the current hypothesis. This eliminates the need for access to the target encoder at inference time and reduces attack cost to a fixed number of forward passes through a small model.

The approach is encoder-agnostic by construction. The embedding vector enters only through AdaLN modulation of layer normalization parameters, so the same architecture works with any embedding model without alignment training or model-specific modifications.

## Architecture

![Architecture](architecture.png)

The system operates in two stages:

1. **Encoding**: Input text is embedded into a 1024-dimensional vector using Qwen3-Embedding
2. **Decoding**: The embedding conditions a masked diffusion language model with AdaLN-Zero, which iteratively unmasks tokens through diffusion sampling

The diffusion process starts from a fully masked sequence and progressively reveals tokens:
- t=1.0: `[MASK] [MASK] [MASK] ...` (fully masked)
- t=0.5: `The quick [MASK] [MASK] fox ...` (partial)
- t=0.0: `The quick brown fox jumps over the lazy dog` (reconstructed)

All tokens refine in parallel based on global embedding context, avoiding autoregressive error accumulation.

## Model Details

- **Backbone**: 8-layer transformer, 768 hidden dim, 12 attention heads
- **Parameters**: 78M trainable (270M total with vocabulary embeddings)
- **Conditioning**: AdaLN modulation at each layer, taking both timestep and embedding as input
- **Training**: 200K steps on 2M C4 samples, log-linear noise schedule with λ=5.0
- **Checkpoint**: 744M (includes EMA weights and optimizer state)

## Setup

Install dependencies:
```bash
pip install fastapi uvicorn torch transformers numpy pyyaml
```

Download the checkpoint (744M):
```bash
# The checkpoint is not in this repo. Place it in ~/checkpoints/
mkdir -p ~/checkpoints
# scp user@host:~/checkpoints/qwen3_best.pt ~/checkpoints/
```

The server expects the checkpoint at `~/checkpoints/qwen3_best.pt`.

## Running the Demo

Start the server:
```bash
python demo_server.py
```

Open `http://localhost:8080` in your browser. Type a sentence, click "Invert", and watch the diffusion process reconstruct the text from its embedding.

## Results

On 32-token sequences from C4:
- **Token accuracy**: 69.7%
- **Cosine similarity**: 0.83
- **Exact match**: 12.3%
- **Inference time**: 150ms per sequence (sequential), 50ms (parallel Euler sampling)

## Privacy Implications

Embedding inversion has evolved from theoretical possibility to practical attack. Organizations deploying embedding-based systems must reassess their threat model: embeddings leak information comparable to the original text and require equivalent protection.

Domain-specific embeddings in medical, financial, and legal applications face amplified risk due to constrained vocabulary and semantic spaces that facilitate inversion.

## Files

- `demo_server.py`: FastAPI server handling encode/decode endpoints
- `index.html`: Interactive web UI with real-time diffusion visualization
- `model.py`: Conditional MDLM with AdaLN conditioning
- `invert.py`: Standalone diffusion sampling for command-line use
- `configs/v2_qwen3.yaml`: Model configuration
- `checkpoints/qwen3_best.pt`: Trained model weights (not in repo, 744M)

## License

Apache 2.0. See LICENSE file.
