"""
Async training loop for text+loudness conditioned rectified flow.

- Prefetches up to 5 batches in the background via dataProc.process_batch()
- Rectified flow objective: predict velocity v = target - noise
- Reports loss every 100 steps
- Dumps a 10-step Euler sample as MP3 every 100 steps

Usage:
    uv run python train.py
"""

import asyncio
import time
from pathlib import Path

import numpy as np
import torch

from dataProc import process_batch, _codec
from perceiver import RectifiedFlowPerceiver, text_to_ids, CHAR_PAD_ID

# ── Constants ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
PREFETCH_DEPTH = 128   # max queued ready-batches
NUM_WORKERS = 8         # concurrent fetch coroutines
LR = 1e-4
N_EULER_SAMPLE = 10
REPORT_EVERY = 100
SAMPLE_DIR = Path("data/samples")

# EnCodec 24kHz: 75 frames per second, 5 seconds = 375 frames
N_TARGET_FRAMES = 375
ENCODEC_DIM = 128
N_LOUDNESS = 50
N_CHARS = 128


# ── Collation ───────────────────────────────────────────────────────

def _text_to_id_tensor(text: str) -> torch.Tensor:
    """Convert a 128-char normalized transcription to a [128] long tensor."""
    ids = text_to_ids(text)
    # Pad or truncate to N_CHARS
    if len(ids) >= N_CHARS:
        ids = ids[:N_CHARS]
    else:
        ids = ids + [CHAR_PAD_ID] * (N_CHARS - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate a list of dicts from process_batch() into GPU tensors.

    Returns dict with:
      - target       : (B, N_TARGET_FRAMES, ENCODEC_DIM)
      - text_ids     : (B, N_CHARS)
      - loudness     : (B, N_LOUDNESS)
    """
    targets = []
    text_ids = []
    loudness_list = []

    for item in batch:
        # encodec_emb is [1, 128, T_frames] — squeeze batch, permute to [T, 128]
        emb = item["encodec_emb"].squeeze(0).permute(1, 0)  # [T_frames, 128]
        # Truncate or pad to exactly N_TARGET_FRAMES
        T = emb.shape[0]
        if T >= N_TARGET_FRAMES:
            emb = emb[:N_TARGET_FRAMES]
        else:
            pad = torch.zeros(N_TARGET_FRAMES - T, ENCODEC_DIM)
            emb = torch.cat([emb, pad], dim=0)
        targets.append(emb)

        text_ids.append(_text_to_id_tensor(item["transcription"]))

        loud = item["loudness"]
        if isinstance(loud, np.ndarray):
            loud = torch.from_numpy(loud).float()
        # Pad or truncate to N_LOUDNESS
        if loud.shape[0] >= N_LOUDNESS:
            loud = loud[:N_LOUDNESS]
        else:
            loud = torch.cat([loud, torch.zeros(N_LOUDNESS - loud.shape[0])])
        loudness_list.append(loud)

    return {
        "target": torch.stack(targets).to(DEVICE),          # (B, 375, 128)
        "text_ids": torch.stack(text_ids).to(DEVICE),       # (B, 128)
        "loudness": torch.stack(loudness_list).to(DEVICE),  # (B, 50)
    }


# ── Prefetch ────────────────────────────────────────────────────────

async def prefetch_worker(queue: asyncio.Queue):
    """Continuously fetch and collate batches, pushing into the queue."""
    while True:
        try:
            raw_batch = await process_batch(BATCH_SIZE)
            if len(raw_batch) == 0:
                continue
            tensors = collate(raw_batch)
            await queue.put(tensors)
        except Exception as e:
            print(f"[prefetch] error: {e}")
            await asyncio.sleep(1.0)


# ── Sampling ────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(
    model: RectifiedFlowPerceiver,
    text_ids: torch.Tensor,   # (1, 128)
    loudness: torch.Tensor,   # (1, 50)
    n_steps: int = N_EULER_SAMPLE,
) -> torch.Tensor:
    """Euler-sample from noise → target in n_steps. Returns (1, T, 128)."""
    model.eval()
    z = torch.randn(1, N_TARGET_FRAMES, ENCODEC_DIM, device=DEVICE)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.tensor([i / n_steps], device=DEVICE)
        v = model(z, t, text_ids, loudness)
        z = z + v * dt

    model.train()
    return z


def save_sample(emb: torch.Tensor, path: str):
    """Decode [1, T, 128] embeddings to audio file via EnCodec."""
    # _codec.decode_continuous expects [1, 128, T]
    emb_for_decode = emb.permute(0, 2, 1).cpu()
    _codec.decode_continuous(emb_for_decode, path)


# ── Training loop ───────────────────────────────────────────────────

async def train_loop(queue: asyncio.Queue):
    """Main training loop — pops batches from the prefetch queue."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    model = RectifiedFlowPerceiver(
        encodec_dim=ENCODEC_DIM,
        n_target_frames=N_TARGET_FRAMES,
        n_loudness=N_LOUDNESS,
        n_chars=N_CHARS,
        dim=512,
        num_heads=8,
        num_context_layers=3,
        num_latent_layers=8,
        n_latents=256,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    running_loss = 0.0

    while True:
        # Pop a pre-fetched batch (blocks until one is ready)
        t_wait = time.perf_counter()
        batch = await queue.get()
        dt_wait = time.perf_counter() - t_wait
        print(f"[train] step {step+1}: got batch (waited {dt_wait:.3f}s, queue_size={queue.qsize()})")

        t_step = time.perf_counter()

        target = batch["target"]       # (B, 375, 128)
        text_ids = batch["text_ids"]   # (B, 128)
        loudness = batch["loudness"]   # (B, 50)
        B = target.shape[0]

        # Rectified flow: sample t, interpolate, predict velocity
        t = torch.rand(B, device=DEVICE)
        x0 = torch.randn_like(target)

        t_broad = t.view(B, 1, 1)
        z_t = (1.0 - t_broad) * x0 + t_broad * target
        v_target = target - x0

        v_pred = model(z_t, t, text_ids, loudness)
        loss = (v_pred - v_target).abs().mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        dt_step = time.perf_counter() - t_step

        step += 1
        running_loss += loss.item()
        print(f"[train] step {step}: loss={loss.item():.6f} fwd+bwd={dt_step:.3f}s")

        if step % REPORT_EVERY == 0:
            avg_loss = running_loss / REPORT_EVERY
            running_loss = 0.0
            print(f"══════ step {step} | avg_loss = {avg_loss:.6f} ══════")

            # Euler-sample one example from this batch
            t_sample = time.perf_counter()
            sample_emb = euler_sample(
                model,
                text_ids[:1],
                loudness[:1],
                n_steps=N_EULER_SAMPLE,
            )
            sample_path = str(SAMPLE_DIR / f"step_{step:06d}.mp3")
            save_sample(sample_emb, sample_path)
            dt_sample = time.perf_counter() - t_sample
            print(f"  → saved sample: {sample_path} (sampling took {dt_sample:.3f}s)")

        # Yield to event loop so prefetch tasks can run
        await asyncio.sleep(0)


# ── Main ────────────────────────────────────────────────────────────

async def main():
    queue: asyncio.Queue = asyncio.Queue(maxsize=PREFETCH_DEPTH)

    # Launch prefetch workers (NUM_WORKERS coroutines filling the queue)
    prefetch_tasks = [
        asyncio.create_task(prefetch_worker(queue))
        for _ in range(NUM_WORKERS)
    ]

    print(f"Training on {DEVICE} | batch_size={BATCH_SIZE} | workers={NUM_WORKERS} | queue_depth={PREFETCH_DEPTH}")
    print("Waiting for first batch...")

    try:
        await train_loop(queue)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        for t in prefetch_tasks:
            t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
