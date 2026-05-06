"""
Residual elastic perceiver for rectified flow on EnCodec latent tokens.

Architecture:
  1. Time t ∈ [0,1] → MLP → prepended as extra token
  2. Cross-attn DOWN: elastic latent bank attends to input tokens
  3. Self-attention in latent space
  4. Cross-attn UP (ELIT-style residual): input attends to latents
  5. Strip the time token, project back to input dim → velocity prediction

All attention layers use RoPE (via torchtune) for positional encoding.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings


# ---------------------------------------------------------------------------
# DenseMLP — the simplest building block
# ---------------------------------------------------------------------------

class DenseMLP(nn.Module):
    """
    ``num_layers`` repetitions of (Linear → LayerNorm → SiLU).
    All layers share the same ``hidden_dim``.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MLP — project up, DenseMLP + skip, project down
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Linear(in→hidden) → [DenseMLP(2 layers) + skip] → Linear(hidden→out).

    The skip connection is:  x' = x + DenseMLP(x)  (in the hidden space).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        num_hidden_blocks: int = 1,
    ):
        super().__init__()
        self.up = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.ModuleList([
            DenseMLP(hidden_dim, num_layers=2)
            for _ in range(num_hidden_blocks)
        ])
        self.down = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        for block in self.blocks:
            x = x + block(x)  # skip connection
        x = self.down(x)
        return x


# ---------------------------------------------------------------------------
# RoPE-enabled attention primitives
# ---------------------------------------------------------------------------

class RoPECrossAttention(nn.Module):
    """
    Pre-norm cross-attention with RoPE on Q and K.

    ``query`` attends to ``kv``.  RoPE is applied to Q (using query
    positions) and K (using kv positions).
    """

    def __init__(self, dim: int, num_heads: int = 8, max_seq_len: int = 2048):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=max_seq_len
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, S_q, _ = q.shape
        S_kv = kv.shape[1]

        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        # Project to [B, S, num_heads, head_dim]
        Q = self.w_q(q_n).view(B, S_q, self.num_heads, self.head_dim)
        K = self.w_k(kv_n).view(B, S_kv, self.num_heads, self.head_dim)
        V = self.w_v(kv_n).view(B, S_kv, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)

        # Transpose for attention: [B, num_heads, S, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(Q, K, V)  # [B, nh, S_q, hd]

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.w_o(out)


class RoPESelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN with residuals. Uses RoPE."""

    def __init__(self, dim: int, num_heads: int = 8, max_seq_len: int = 2048):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=max_seq_len
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # --- Self-attention with RoPE ---
        normed = self.norm1(x)
        Q = self.w_q(normed).view(B, S, self.num_heads, self.head_dim)
        K = self.w_k(normed).view(B, S, self.num_heads, self.head_dim)
        V = self.w_v(normed).view(B, S, self.num_heads, self.head_dim)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        attn_out = self.w_o(attn_out)

        x = x + attn_out

        # --- FFN with residual ---
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Rectified-Flow Perceiver
# ---------------------------------------------------------------------------

class RectifiedFlowPerceiver(nn.Module):
    """
    Elastic perceiver that predicts the velocity field for rectified flow.

    Forward signature::

        v = model(z_t, t)
        # z_t : (B, seq_len, input_dim)  — noisy latent tokens
        # t   : (B,)                     — diffusion time in [0, 1]
        # v   : (B, seq_len, input_dim)  — predicted velocity

    Architecture
    ------------
    1. **Time conditioning**: ``t`` is projected via an MLP to a single
       ``dim``-dimensional token and prepended to the input sequence.
    2. **Input projection**: Linear(input_dim → dim).
    3. **Cross-attn DOWN**: learnable latent bank (``max_latents × dim``)
       attends to the (time + input) tokens — elastic compute by slicing.
    4. **Self-attention** layers in latent space.
    5. **Cross-attn UP** (ELIT-style residual):
       ``cross_out = cross_attn(q=x, kv=latents)``
       ``z = LayerNorm(cross_out) + x``
       ``fused = MLP(z) + z``
    6. Strip the first (time) token → last ``seq_len`` vectors.
    7. Output MLP back to ``input_dim`` → velocity prediction.

    All attention uses RoPE for positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 512,
        num_heads: int = 8,
        num_latent_layers: int = 6,
        max_latents: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.max_latents = max_latents
        self.head_dim = dim // num_heads

        # --- Time conditioning ---
        self.time_mlp = MLP(
            input_size=1,
            output_size=dim,
            hidden_dim=dim,
            num_hidden_blocks=1,
        )

        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, dim)

        # --- Learnable latent bank (sliced for elastic compute) ---
        self.latents = nn.Parameter(torch.randn(max_latents, dim) * 0.02)

        # --- Cross-attn DOWN: latents attend to input tokens ---
        self.cross_attn_down = RoPECrossAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len
        )

        # --- Self-attention in latent space ---
        self.latent_layers = nn.ModuleList([
            RoPESelfAttentionBlock(
                dim=dim, num_heads=num_heads, max_seq_len=max_seq_len
            )
            for _ in range(num_latent_layers)
        ])

        # --- Cross-attn UP (ELIT-style) ---
        self.cross_attn_up = RoPECrossAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len
        )
        self.post_cross_norm = nn.LayerNorm(dim)
        self.post_cross_mlp = MLP(
            input_size=dim,
            output_size=dim,
            hidden_dim=dim,
            num_hidden_blocks=1,
        )

        # --- Output projection: dim → input_dim (velocity) ---
        self.output_mlp = MLP(
            input_size=dim,
            output_size=input_dim,
            hidden_dim=dim,
            num_hidden_blocks=1,
        )

    def forward(
        self,
        z_t: torch.Tensor,        # (B, seq_len, input_dim) — noisy latents
        t: torch.Tensor,           # (B,) — time in [0, 1]
        n_latents: int | None = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        v : (B, seq_len, input_dim) — predicted velocity for rectified flow.
        """
        B, seq_len, _ = z_t.shape

        # (1) Time token: (B,) → (B, 1, 1) → MLP → (B, 1, dim)
        t_tok = self.time_mlp(t.view(B, 1, 1))  # (B, 1, dim)

        # (2) Project input tokens
        x = self.input_proj(z_t)  # (B, seq_len, dim)

        # Prepend time token → (B, 1 + seq_len, dim)
        x = torch.cat([t_tok, x], dim=1)

        # (3) Elastic latents
        if n_latents is None:
            n_latents = self.max_latents
        n_latents = max(1, min(n_latents, self.max_latents))
        latents = self.latents[:n_latents].unsqueeze(0).expand(B, -1, -1)

        # Cross-attn DOWN + residual
        latents = self.cross_attn_down(latents, x) + latents

        # (4) Self-attention in latent space
        for layer in self.latent_layers:
            latents = layer(latents)

        # (5) Cross-attn UP — ELIT-style residual
        cross_out = self.cross_attn_up(x, latents)
        z = self.post_cross_norm(cross_out) + x
        fused = self.post_cross_mlp(z) + z  # (B, 1 + seq_len, dim)

        # (6) Strip time token → keep last seq_len vectors
        fused = fused[:, 1:, :]  # (B, seq_len, dim)

        # (7) Output projection → velocity
        v = self.output_mlp(fused)  # (B, seq_len, input_dim)
        return v
