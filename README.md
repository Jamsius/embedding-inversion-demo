# Embedding Inversion via Masked Diffusion

Diffusion-based embedding inversion using Qwen3-Embedding and a conditional masked diffusion language model (MDLM). Given an embedding vector, reconstruct the original text through iterative unmasking.

## Architecture

![Architecture](architecture.png)

The system operates in two stages:

1. **Encoding**: Input text is embedded into a 1024-dimensional vector using Qwen3-Embedding
2. **Decoding**: The embedding conditions an MDLM with AdaLN-Zero conditioning, which iteratively unmasks tokens through diffusion sampling

The diffusion process starts from a fully masked sequence and progressively reveals tokens conditioned on the embedding:
- t=0: `[MASK] [MASK] [MASK] ...`
- t=k: `The quick [MASK] [MASK] fox ...` (partial)
- t=T: `The quick brown fox jumps over the lazy dog` (reconstructed)

## Method

The model uses:
- **MDLM backbone**: Masked diffusion for discrete token generation (Sahoo et al., NeurIPS 2024)
- **AdaLN-Zero conditioning**: Scale and shift modulation based on embedding vector (Peebles & Xie, ICCV 2023)
- **Qwen3-Embedding**: 1024-d sentence embeddings

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

## Files

- `demo_server.py`: FastAPI server handling encode/decode endpoints
- `index.html`: Interactive web UI
- `model.py`: MDLM transformer with AdaLN-Zero conditioning
- `invert.py`: Diffusion sampling logic
- `configs/v2_qwen3.yaml`: Model configuration
- `checkpoints/qwen3_best.pt`: Trained model weights (not in repo)

## References

- Sahoo, Prasanna, et al. "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024.
- Peebles, William, and Saining Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
