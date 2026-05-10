"""
Text + loudness conditioned perceiver for rectified flow on EnCodec latents.

Architecture:
  1. Time t ∈ [0,1] → MLP → 1 conditioning token
  2. Transcription (128 chars) → char_emb + learned char_type_emb → 128 tokens
  3. Loudness (50 scalars) → per-value Linear→LN→MLP + learned loudness_type_emb → 50 tokens
  4. Noisy signal z_t [T_target, 128] → MLP(128→dim) + learned signal_type_emb → T_target tokens
  5. Concatenate all → 3 layers RoPE self-attention (context processing)
  6. Cross-attn DOWN: 256 learned latents attend to context
  7. 8 layers RoPE self-attention in latent space
  8. Cross-attn UP: learned decoder array [T_target, dim] attends to latents
  9. Output MLP → velocity [B, T_target, 128]

Trained with rectified flow objective against EnCodec continuous embeddings.

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
# Rectified-Flow Perceiver
# ---------------------------------------------------------------------------

class RectifiedFlowPerceiver(nn.Module):
    """
    Text + loudness conditioned perceiver that predicts the velocity field
    for rectified flow on EnCodec latent tokens.

    Forward signature::

        v = model(z_t, t, transcription_ids, loudness)

    Parameters
    ----------
    z_t                : (B, n_target_frames, encodec_dim)  — noisy flow sample
    t                  : (B,)                               — flow time in [0, 1]
    transcription_ids  : (B, 128)                           — char IDs (fixed length)
    loudness           : (B, 50)                            — per-100ms RMS values

    Returns
    -------
    v : (B, n_target_frames, encodec_dim) — predicted velocity
    """

    def __init__(
        self,
        encodec_dim: int = 128,
        n_target_frames: int = 375,
        n_loudness: int = 50,
        n_chars: int = 128,
        dim: int = 512,
        num_heads: int = 8,
        num_context_layers: int = 3,
        num_latent_layers: int = 8,
        n_latents: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.encodec_dim = encodec_dim
        self.n_target_frames = n_target_frames
        self.dim = dim

        # ── Time conditioning ───────────────────────────────────────
        self.time_mlp = MLP(
            input_size=1, output_size=dim, hidden_dim=dim, num_hidden_blocks=1,
        )

        # ── Transcription embeddings ────────────────────────────────
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, dim, padding_idx=CHAR_PAD_ID)
        self.char_type_emb = nn.Parameter(torch.randn(dim) * 0.02)

        # ── Loudness embeddings ─────────────────────────────────────
        self.loudness_proj = nn.Linear(1, dim)
        self.loudness_ln = nn.LayerNorm(dim)
        self.loudness_mlp = MLP(
            input_size=dim, output_size=dim, hidden_dim=dim, num_hidden_blocks=1,
        )
        self.loudness_type_emb = nn.Parameter(torch.randn(dim) * 0.02)

        # ── Noisy signal (z_t) embedding ────────────────────────────
        self.signal_mlp = MLP(
            input_size=encodec_dim, output_size=dim, hidden_dim=dim,
            num_hidden_blocks=1,
        )
        self.signal_type_emb = nn.Parameter(torch.randn(dim) * 0.02)

        # ── Context self-attention ──────────────────────────────────
        self.context_layers = nn.ModuleList([
            RoPESelfAttentionBlock(
                dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
            )
            for _ in range(num_context_layers)
        ])

        # ── Learnable latent bank ───────────────────────────────────
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)

        # ── Cross-attn DOWN: latents attend to context ──────────────
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

        # ── Learned decoder array ───────────────────────────────────
        self.decoder_tokens = nn.Parameter(
            torch.randn(n_target_frames, dim) * 0.02
        )

        # ── Cross-attn UP: decoder attends to latents ──────────────
        self.cross_attn_up = RoPECrossAttention(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
        )

        # ── Output projection → velocity ───────────────────────────
        self.output_mlp = MLP(
            input_size=dim, output_size=encodec_dim, hidden_dim=dim,
            num_hidden_blocks=1,
        )

    def forward(
        self,
        z_t: torch.Tensor,                # (B, n_target_frames, encodec_dim)
        t: torch.Tensor,                   # (B,)
        transcription_ids: torch.Tensor,   # (B, 128)
        loudness: torch.Tensor,            # (B, 50)
    ) -> torch.Tensor:
        B = z_t.shape[0]

        # ① Time token: (B, 1, dim)
        t_tok = self.time_mlp(t.view(B, 1, 1))

        # ② Transcription tokens: (B, 128, dim)
        char_tok = self.char_emb(transcription_ids) + self.char_type_emb

        # ③ Loudness tokens: (B, 50, dim)
        loud_tok = self.loudness_proj(loudness.unsqueeze(-1))  # (B, 50, dim)
        loud_tok = self.loudness_ln(loud_tok)
        loud_tok = self.loudness_mlp(loud_tok) + self.loudness_type_emb

        # ④ Noisy signal tokens: (B, n_target_frames, dim)
        signal_tok = self.signal_mlp(z_t) + self.signal_type_emb

        # ⑤ Concatenate all context tokens
        context = torch.cat([t_tok, char_tok, loud_tok, signal_tok], dim=1)

        # ⑥ Context self-attention (3 layers)
        for layer in self.context_layers:
            context = layer(context)

        # ⑦ Cross-attn DOWN: latents attend to context
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents = self.cross_attn_down(latents, context) + latents

        # ⑧ Self-attention in latent space (8 layers)
        for layer in self.latent_layers:
            latents = layer(latents)

        # ⑨ Cross-attn UP: decoder tokens attend to latents
        decoder = self.decoder_tokens.unsqueeze(0).expand(B, -1, -1)
        decoded = self.cross_attn_up(decoder, latents)

        # ⑩ Output → velocity
        v = self.output_mlp(decoded)  # (B, n_target_frames, encodec_dim)
        return v
