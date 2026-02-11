# Embedding Inversion via Conditional Masked Diffusion

Text embeddings are widely assumed to be safe, irreversible representations. This project demonstrates otherwise: given only an embedding vector, we reconstruct the original text using conditional masked diffusion.

## How It Works

Existing inversion methods (Vec2Text, ALGEN, Zero2Text) generate tokens autoregressively and require iterative re-embedding through the target encoder. This creates two bottlenecks: attack cost scales with correction iterations, and left-to-right generation accumulates errors with no mechanism to revise earlier tokens.

We take a different approach: **embedding inversion as conditional masked diffusion**. Starting from a fully masked sequence, a denoising model reveals tokens at all positions in parallel, conditioned on the target embedding via adaptive layer normalization (AdaLN-Zero). Each denoising step refines all positions simultaneously using global context, without ever re-embedding the current hypothesis.

The approach is encoder-agnostic by construction. The embedding vector enters only through AdaLN modulation of layer normalization parameters, so the same architecture applies to any embedding model without alignment training or architecture-specific modifications.

### Architecture

![Architecture](architecture.png)

```
Input text --> Encoder (Qwen3-Embedding-0.6B) --> e in R^1024
                                                      |
                                                      v
                                              Projection MLP
                                                      |
                                                      v
                                                c in R^768
                                                      |
[MASK][MASK]...[MASK] --> Token Embed --> 8x Transformer Blocks (AdaLN-Zero) --> Logits --> Predicted Tokens
       t=1.0                                                                                    |
                                                                                          Iterative
                                                                                          Denoising
                                                                                                |
                                                                                                v
                                                                                    Reconstructed Text
                                                                                          t=0.0
```

Each transformer block applies AdaLN-Zero conditioning: the conditioning vector produces per-layer scale, shift, and gate parameters, all initialized to zero so the model starts as a vanilla transformer and gradually learns to use the embedding signal.

### Results

On 32-token sequences, the 78M-parameter model achieves:
- 78% token accuracy
- 0.83 cosine similarity between original and reconstructed embeddings

## Live Demo

https://embedding-inversion-demo.jina.ai

## Setup

```bash
pip install -r requirements.txt
```

Place the checkpoint at `checkpoints/qwen3_best.pt` (not included in repo).

```bash
python demo_server.py
```

Server starts on port 8080.

## Files

| File | Description |
|------|-------------|
| demo_server.py | FastAPI server with encode/decode endpoints |
| model.py | Conditional MDLM architecture |
| invert.py | Diffusion sampling strategies |
| index.html | Interactive demo frontend |
| configs/v2_qwen3.yaml | Model configuration |

## License

Apache 2.0
