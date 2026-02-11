"""
Conditional Masked Diffusion Language Model (CMDLM) for embedding inversion.

Architecture: mmBERT backbone with AdaLN-Zero conditioning on jina-embeddings-v3 vectors.
Input: masked token sequence [B, L] + embedding condition [B, 1024]
Output: logits [B, L, V] predicting original tokens at all positions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (DiT-style).
    
    Output: norm(x) * (1 + scale) + shift
    Gate: x = x + alpha * block_output (alpha initialized to 0)
    
    All conditioning projections initialized to zero so the model starts
    as identity (vanilla mmBERT) and gradually learns to use conditioning.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Project condition to (scale, shift, alpha)
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        # Zero initialization - model starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        """
        Args:
            x: [B, L, hidden] input
            cond: [B, cond_dim] conditioning vector
        
        Returns:
            normalized: norm(x) * (1 + scale) + shift
            alpha: [B, 1, hidden] gate for residual (to be used as x = x + alpha * block_output)
        """
        # cond: [B, cond_dim] -> [B, 3*hidden]
        params = self.proj(cond).unsqueeze(1)  # [B, 1, 3*hidden]
        scale, shift, alpha = params.chunk(3, dim=-1)  # each [B, 1, hidden]
        normalized = self.norm(x) * (1 + scale) + shift
        return normalized, alpha


class ModernBertLayerWithAdaLN(nn.Module):
    """
    Wraps a pretrained ModernBERT layer with AdaLN-Zero conditioning.
    
    Architecture:
        - Replace layer's internal norms with AdaLN-Zero
        - Add gated residuals with alpha initialized to 0
    """

    def __init__(self, pretrained_layer, hidden_dim, cond_dim):
        super().__init__()
        self.pretrained_layer = pretrained_layer
        
        # AdaLN-Zero conditioning  parameters (scale, shift, gate)
        # We project cond to get 6*hidden: (scale_attn, shift_attn, alpha_attn, scale_mlp, shift_mlp, alpha_mlp)
        self.cond_proj = nn.Linear(cond_dim, 6 * hidden_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        
        # Store reference to original norms (we'll replace their functionality)
        self.attn_norm = pretrained_layer.attn_norm  
        self.mlp_norm = pretrained_layer.mlp_norm

    def forward(self, hidden_states, cond, position_embeddings):
        """
        Args:
            hidden_states: [B, L, hidden]
            cond: [B, cond_dim] conditioning vector
            position_embeddings: (cos, sin) tuple for RoPE
        
        Returns:
            hidden_states: [B, L, hidden]
        """
        # Get all AdaLN-Zero parameters at once
        adaln_params = self.cond_proj(cond).unsqueeze(1)  # [B, 1, 6*hidden]
        scale_attn, shift_attn, alpha_attn, scale_mlp, shift_mlp, alpha_mlp = \
            adaln_params.chunk(6, dim=-1)  # each [B, 1, hidden]
        
        # Attention block
        # Apply AdaLN-Zero norm (replace attn_norm)
        if isinstance(self.attn_norm, nn.Identity):
            # Layer 0 has no norm
            normed_attn = hidden_states * (1 + scale_attn) + shift_attn
        else:
            # Use pretrained norm but apply AdaLN conditioning
            normed_attn = self.attn_norm(hidden_states) * (1 + scale_attn) + shift_attn
        
        # Attention (returns tuple (output, attention_weights) but we only need output)
        attn_out = self.pretrained_layer.attn(
            normed_attn,
            attention_mask=None,
            sliding_window_mask=None,
            position_ids=None,
            position_embeddings=position_embeddings
        )
        # attn might return a tuple, take first element
        attn_output = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        
        # Gated residual: alpha_attn is [B, 1, hidden], broadcasts to [B, L, hidden]
        hidden_states = hidden_states + (alpha_attn * attn_output)
        
        # MLP block
        # Apply AdaLN-Zero norm (replace mlp_norm)
        normed_mlp = self.mlp_norm(hidden_states) * (1 + scale_mlp) + shift_mlp
        
        # MLP
        mlp_output = self.pretrained_layer.mlp(normed_mlp)
        
        # Gated residual: alpha_mlp is [B, 1, hidden], broadcasts to [B, L, hidden]
        hidden_states = hidden_states + (alpha_mlp * mlp_output)
        
        return hidden_states


class ConditionalMDLM(nn.Module):
    """
    Conditional Masked Diffusion Language Model with mmBERT backbone.

    Takes masked token sequences and a conditioning embedding vector,
    predicts the original tokens at all positions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        mc = config["model"]

        self.vocab_size = mc["vocab_size"]
        self.hidden_dim = mc["hidden_dim"]
        self.max_seq_len = mc["max_seq_len"]
        self.mask_token_id = mc["mask_token_id"]

        # Load pretrained mmBERT model
        pretrained_model = mc.get("pretrained_token_embeddings")
        if not pretrained_model:
            raise ValueError("pretrained_token_embeddings must be specified (e.g., jhu-clsp/mmBERT-base)")
        
        print(f"Loading pretrained mmBERT from {pretrained_model}...")
        from transformers import AutoModel
        mmbert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        
        # Extract components
        pretrained_embeddings = mmbert.embeddings.tok_embeddings
        self.embed_norm = mmbert.embeddings.norm
        self.embed_drop = mmbert.embeddings.drop
        # rotary_emb handled internally by ModernBert
        
        # Extend token embeddings if vocab_size is larger (e.g., for mask token)
        pretrained_vocab = pretrained_embeddings.weight.shape[0]
        if self.vocab_size > pretrained_vocab:
            print(f"Extending embeddings from {pretrained_vocab} to {self.vocab_size} (adding mask token)")
            self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim)
            # Copy pretrained weights
            self.token_embed.weight.data[:pretrained_vocab].copy_(pretrained_embeddings.weight.data)
            # Initialize extra tokens (mask token) with small random values
            nn.init.normal_(self.token_embed.weight.data[pretrained_vocab:], std=0.02)
        else:
            self.token_embed = pretrained_embeddings
        
        # Freeze token embeddings if specified
        self.freeze_embeddings = mc.get("freeze_token_embeddings", False)
        if self.freeze_embeddings:
            self.token_embed.weight.requires_grad_(False)
            print(f"Frozen token embeddings ({self.vocab_size} x {self.hidden_dim})")
        
        # Project conditioning embedding to internal dim
        cond_dim = mc["embedding_cond_dim"]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Wrap pretrained transformer layers with AdaLN-Zero
        print(f"Wrapping {len(mmbert.layers)} mmBERT layers with AdaLN-Zero conditioning...")
        self.layers = nn.ModuleList([
            ModernBertLayerWithAdaLN(layer, self.hidden_dim, self.hidden_dim)
            for layer in mmbert.layers
        ])
        
        # Final layer norm with AdaLN-Zero
        self.final_adaln = AdaLNZero(self.hidden_dim, self.hidden_dim)
        
        # Output projection
        self.tie_weights = mc.get("tie_weights", True)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        
        if self.tie_weights:
            # Tie with token embeddings
            self.output_proj.weight = self.token_embed.weight
            if self.freeze_embeddings:
                # If embeddings are frozen and tied, output_proj is also frozen
                print("Weight tying: output_proj shares weights with token_embed (frozen)")
            else:
                print("Weight tying: output_proj shares weights with token_embed")
        else:
            # Initialize from token embeddings (already extended to vocab_size)
            self.output_proj.weight.data.copy_(self.token_embed.weight.data)
            print("output_proj initialized from token embeddings (independent, trainable)")
        
        # RoPE position embeddings handled internally by ModernBert
        
        # Clean up
        del mmbert
        
        # Gradient checkpointing (disabled by default)
        self.use_checkpoint = False
        
        # Count parameters
        total, trainable = self.count_params()
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_ids, cond_embedding, padding_mask=None):
        """
        Args:
            input_ids: [B, L] token ids (with some positions replaced by mask_token_id)
            cond_embedding: [B, 1024] target embedding vector
            padding_mask: [B, L] True where padding (to be ignored)

        Returns:
            logits: [B, L, V] prediction logits for all positions
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Token embeddings
        hidden_states = self.token_embed(input_ids)  # [B, L, hidden]
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_drop(hidden_states)
        
        # Project conditioning
        cond = self.cond_proj(cond_embedding)  # [B, hidden]
        
        # RoPE handled internally
        position_embeddings = None
        
        # Apply transformer layers with AdaLN-Zero conditioning
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                hidden_states = torch_checkpoint(
                    layer, hidden_states, cond, position_embeddings,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cond, position_embeddings)
        
        # Final norm with AdaLN-Zero
        hidden_states, _ = self.final_adaln(hidden_states, cond)
        
        # Output projection
        logits = self.output_proj(hidden_states)  # [B, L, V]

        return logits

    def forward_hidden(self, input_ids, cond_embedding, padding_mask=None):
        """Same as forward but returns hidden states before output_proj."""
        B, L = input_ids.shape
        device = input_ids.device

        # Token embeddings
        hidden_states = self.token_embed(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_drop(hidden_states)
        
        # Project conditioning
        cond = self.cond_proj(cond_embedding)
        
        # RoPE handled internally
        position_embeddings = None
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cond, position_embeddings)
        
        # Final norm
        hidden_states, _ = self.final_adaln(hidden_states, cond)
        
        return hidden_states  # [B, L, hidden]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def apply_mask(token_ids, mask_token_id, padding_mask=None):
    """
    Apply random masking for MDLM training.

    For each sample in batch, randomly choose a mask ratio in [0.1, 1.0],
    then mask that fraction of non-padding tokens.

    Returns:
        masked_ids: [B, L] with some tokens replaced by mask_token_id
        target_mask: [B, L] boolean, True at positions that were masked (loss targets)
        mask_ratio: [B, 1] the mask ratio used for each sample
    """
    B, L = token_ids.shape
    device = token_ids.device

    # Random mask ratio per sample
    # Log-linear noise schedule (MDLM paper: better than uniform)
    u = torch.rand(B, 1, device=device)
    eps = 1e-3
    mask_ratio = 1 - (1 - eps) ** u  # log-linear, range ~[eps, 1.0]
    mask_ratio = mask_ratio.clamp(min=0.1, max=1.0)  # keep min 0.1 for stability

    # Random scores for each position
    rand_scores = torch.rand(B, L, device=device)

    # Don't mask padding positions
    if padding_mask is not None:
        rand_scores[padding_mask] = 2.0  # never mask padding

    # Mask positions where rand < ratio
    target_mask = rand_scores < mask_ratio  # [B, L]

    # Apply mask
    masked_ids = token_ids.clone()
    masked_ids[target_mask] = mask_token_id

    return masked_ids, target_mask, mask_ratio  # mask_ratio: [B, 1]
