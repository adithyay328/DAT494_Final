"""
Text-conditioned residual elastic perceiver for rectified flow.

Architecture:
  1. Time t ∈ [0,1] → MLP → 1 conditioning token
  2. Text chars → char_emb + segment_emb → text conditioning tokens
  3. Context audio → proj + segment_emb → context conditioning tokens
  4. Noisy target → proj + segment_emb → target query tokens
  5. Cross-attn DOWN: latents attend to ALL tokens (time+text+ctx+target)
  6. Self-attention × N in latent space
  7. Cross-attn UP (ELIT residual): only TARGET tokens attend to latents
  8. Output MLP → velocity for target tokens only

Segment embedding IDs:
  0 = prior text      1 = upcoming text
  2 = context audio   3 = target audio

Character vocabulary (37 tokens):
  a-z = 0..25,  0-9 = 26..35,  <pad> = 36

All attention uses RoPE (via torchtune) for positional encoding.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings

# ── Character vocabulary ────────────────────────────────────────────
CHAR_VOCAB_SIZE = 37  # a-z (26) + 0-9 (10) + pad (1)
CHAR_PAD_ID = 36

SEG_PRIOR_TEXT = 0
SEG_UPCOMING_TEXT = 1
SEG_CTX_AUDIO = 2
SEG_TGT_AUDIO = 3


def text_to_ids(text: str) -> list[int]:
    """Strip non-alphanumeric, lowercase, convert to char IDs.

    a-z → 0..25, 0-9 → 26..35.  Everything else is dropped.
    """
    ids: list[int] = []
    for ch in text.lower():
        if "a" <= ch <= "z":
            ids.append(ord(ch) - ord("a"))
        elif "0" <= ch <= "9":
            ids.append(26 + ord(ch) - ord("0"))
    return ids


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
            x = x + block(x)
        x = self.down(x)
        return x


# ---------------------------------------------------------------------------
# RoPE-enabled attention primitives
# ---------------------------------------------------------------------------

class RoPECrossAttention(nn.Module):
    """Pre-norm cross-attention with RoPE on Q and K."""

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

        Q = self.w_q(q_n).view(B, S_q, self.num_heads, self.head_dim)
        K = self.w_k(kv_n).view(B, S_kv, self.num_heads, self.head_dim)
        V = self.w_v(kv_n).view(B, S_kv, self.num_heads, self.head_dim)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.w_o(out)


class RoPESelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN with residuals.  Uses RoPE."""

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
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Rectified-Flow Perceiver (text-conditioned, target-only UP)
# ---------------------------------------------------------------------------

class RectifiedFlowPerceiver(nn.Module):
    """
    Text-conditioned elastic perceiver that predicts the velocity field
    for rectified flow on EnCodec latent tokens.

    Forward signature::

        v = model(z_t, t, context_audio, text_ids, text_segments)

    Parameters
    ----------
    z_t            : (B, target_len, input_dim)  — noisy target audio
    t              : (B,)                        — flow time in [0, 1]
    context_audio  : (B, ctx_len, input_dim)     — clean context audio
    text_ids       : (B, text_len)               — character indices
    text_segments  : (B, text_len)               — 0=prior, 1=upcoming

    Returns
    -------
    v : (B, target_len, input_dim) — predicted velocity (target only)
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 256,
        num_heads: int = 8,
        num_latent_layers: int = 12,
        max_latents: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.max_latents = max_latents

        # ── Time conditioning ───────────────────────────────────────
        self.time_mlp = MLP(
            input_size=1, output_size=dim, hidden_dim=dim, num_hidden_blocks=1,
        )

        # ── Text embeddings ─────────────────────────────────────────
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, dim, padding_idx=CHAR_PAD_ID)

        # ── Segment embeddings (4-way) ──────────────────────────────
        # 0=prior_text, 1=upcoming_text, 2=ctx_audio, 3=tgt_audio
        self.segment_emb = nn.Embedding(4, dim)

        # ── Audio projection ────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, dim)

        # ── Learnable latent bank ───────────────────────────────────
        self.latents = nn.Parameter(torch.randn(max_latents, dim) * 0.02)

        # ── Cross-attn DOWN: latents attend to full input ───────────
        self.cross_attn_down = RoPECrossAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
        )

        # ── Self-attention in latent space ──────────────────────────
        self.latent_layers = nn.ModuleList([
            RoPESelfAttentionBlock(
                dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
            )
            for _ in range(num_latent_layers)
        ])

        # ── Cross-attn UP (target-only, ELIT-style residual) ───────
        self.cross_attn_up = RoPECrossAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
        )
        self.post_cross_norm = nn.LayerNorm(dim)
        self.post_cross_mlp = MLP(
            input_size=dim, output_size=dim, hidden_dim=dim,
            num_hidden_blocks=1,
        )

        # ── Output projection → velocity ───────────────────────────
        self.output_mlp = MLP(
            input_size=dim, output_size=input_dim, hidden_dim=dim,
            num_hidden_blocks=1,
        )

    def forward(
        self,
        z_t: torch.Tensor,            # (B, target_len, input_dim)
        t: torch.Tensor,               # (B,)
        context_audio: torch.Tensor,   # (B, ctx_len, input_dim)
        text_ids: torch.Tensor,        # (B, text_len)
        text_segments: torch.Tensor,   # (B, text_len)  — 0 or 1
        n_latents: int | None = None,
    ) -> torch.Tensor:
        B = z_t.shape[0]

        # ① Time token
        t_tok = self.time_mlp(t.view(B, 1, 1))  # (B, 1, dim)

        # ② Text tokens: char_emb + segment_emb
        text_tok = (
            self.char_emb(text_ids)
            + self.segment_emb(text_segments)
        )  # (B, text_len, dim)

        # ③ Context audio tokens: proj + segment_emb[2]
        ctx_tok = (
            self.input_proj(context_audio)
            + self.segment_emb.weight[SEG_CTX_AUDIO]
        )  # (B, ctx_len, dim)

        # ④ Target audio tokens: proj + segment_emb[3]
        tgt_tok = (
            self.input_proj(z_t)
            + self.segment_emb.weight[SEG_TGT_AUDIO]
        )  # (B, target_len, dim)

        # ⑤ Full input for cross-attn DOWN
        full_input = torch.cat([t_tok, text_tok, ctx_tok, tgt_tok], dim=1)

        # Elastic latents
        if n_latents is None:
            n_latents = self.max_latents
        n_latents = max(1, min(n_latents, self.max_latents))
        latents = self.latents[:n_latents].unsqueeze(0).expand(B, -1, -1)

        # Cross-attn DOWN + residual
        latents = self.cross_attn_down(latents, full_input) + latents

        # ⑥ Self-attention in latent space
        for layer in self.latent_layers:
            latents = layer(latents)

        # ⑦ Cross-attn UP — ONLY target tokens as queries
        cross_out = self.cross_attn_up(tgt_tok, latents)
        z = self.post_cross_norm(cross_out) + tgt_tok   # residual to noisy target
        fused = self.post_cross_mlp(z) + z               # second residual

        # ⑧ Output projection → velocity (target only)
        v = self.output_mlp(fused)  # (B, target_len, input_dim)
        return v
