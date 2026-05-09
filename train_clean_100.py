"""
Text-conditioned rectified-flow training on LibriTTS-R train-clean-100.

Requires: run ``uv run python preprocess.py`` first to EnCodec-encode all
wav files and build the manifest.

- Dataset loads pre-encoded .pt embeddings and transcribes each 1-second
  chunk via Gemini on the fly (I/O-bound, parallelised across DataLoader
  workers).
- Trains rectified flow: 5 s context + text → predict next 1 s
- Every epoch: Euler-sample 10 clips, decode, and save as audio

Usage:
    uv run python train_clean_100.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from codec import Codec
from perceiver import (
    CHAR_PAD_ID,
    SEG_PRIOR_TEXT,
    SEG_UPCOMING_TEXT,
    RectifiedFlowPerceiver,
    text_to_ids,
)
from transcribe import transcribe_audio_bytes

# ── Constants ───────────────────────────────────────────────────────
FPS = 75  # EnCodec frame rate
CHUNK_FRAMES = 1 * FPS      # 75   (1-second chunk)
CTX_CHUNKS = 5               # 5 chunks = 5 s context
TGT_CHUNKS = 1               # 1 chunk  = 1 s target
MIN_CHUNKS = CTX_CHUNKS + TGT_CHUNKS  # 6 chunks = 6 s minimum

CONTEXT_FRAMES = CTX_CHUNKS * CHUNK_FRAMES   # 375
TARGET_FRAMES = TGT_CHUNKS * CHUNK_FRAMES    # 75

CACHE_DIR = Path("data/encodec_cache")
SAMPLE_DIR = Path("data/samples")
MANIFEST_PATH = CACHE_DIR / "manifest.json"

BATCH_SIZE = 8
NUM_WORKERS = 10
EPOCHS = 50
LR = 1e-4
N_EULER = 100
PRINT_EVERY = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sample rate that matches EnCodec (24 kHz)
TARGET_SR = 24_000


# ═══════════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════════
class LibriTTSFlowDataset(Dataset):
    """
    Returns (context_audio, target_audio, text_ids, text_segments).

    Each training sample picks 6 consecutive 1-s chunks from one utterance:
      - chunks [i..i+4] → context audio (375 frames) + prior text
      - chunk  [i+5]    → target audio  (75 frames)  + upcoming text

    Text is obtained by sending each 1-second audio chunk to Gemini
    at load time — parallelised across DataLoader workers.
    """

    def __init__(self):
        if not MANIFEST_PATH.exists():
            print(
                "ERROR: manifest not found. Run `uv run python preprocess.py` first.",
                file=sys.stderr,
            )
            sys.exit(1)

        manifest = json.loads(MANIFEST_PATH.read_text())

        # Keep only entries with enough chunks (≥6 s)
        self.entries: list[dict] = [
            v for v in manifest.values()
            if v.get("n_chunks", 0) >= MIN_CHUNKS
        ]
        print(f"Dataset: {len(self.entries)} usable utterances "
              f"(≥{MIN_CHUNKS} chunks) out of {len(manifest)} total.")

    def __len__(self):
        return len(self.entries)

    def _transcribe_chunk(self, wav: torch.Tensor, sr: int) -> str:
        """Encode a 1-s waveform chunk to mp3 bytes, send to Gemini."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                torchaudio.save(tmp.name, wav, sr, format="mp3")
                audio_bytes = Path(tmp.name).read_bytes()
            return transcribe_audio_bytes(audio_bytes, media_type="audio/mpeg")
        except Exception as e:
            # Transcription failure is non-fatal — return empty string
            return ""

    def __getitem__(self, idx):
        entry = self.entries[idx]
        emb = torch.load(entry["pt_path"], weights_only=True)  # [T, 128]

        n_chunks = entry["n_chunks"]

        # Pick random start chunk
        max_start = n_chunks - MIN_CHUNKS
        start = torch.randint(0, max(1, max_start + 1), (1,)).item()

        # ── Audio embeddings ─────────────────────────────────────
        ctx_s = start * CHUNK_FRAMES
        ctx_e = (start + CTX_CHUNKS) * CHUNK_FRAMES
        context = emb[ctx_s:ctx_e]  # [375, 128]

        tgt_s = (start + CTX_CHUNKS) * CHUNK_FRAMES
        tgt_e = (start + CTX_CHUNKS + TGT_CHUNKS) * CHUNK_FRAMES
        target = emb[tgt_s:tgt_e]  # [75, 128]

        # ── Per-chunk transcription via Gemini ───────────────────
        wav, sr = torchaudio.load(entry["wav_path"])
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        chunk_samples = int(1.0 * TARGET_SR)  # 24000 samples per 1-s chunk

        prior_texts = []
        for i in range(CTX_CHUNKS):
            c = start + i
            s = c * chunk_samples
            e = s + chunk_samples
            chunk_wav = wav[:, s:e]
            prior_texts.append(self._transcribe_chunk(chunk_wav, TARGET_SR))

        # Target chunk text
        tgt_c = start + CTX_CHUNKS
        tgt_wav = wav[:, tgt_c * chunk_samples : (tgt_c + 1) * chunk_samples]
        upcoming_text = self._transcribe_chunk(tgt_wav, TARGET_SR)

        prior_text = " ".join(prior_texts)

        # ── Text → ids + segment labels ──────────────────────────
        prior_ids = text_to_ids(prior_text)
        upcoming_ids = text_to_ids(upcoming_text)

        all_ids = prior_ids + upcoming_ids
        all_segs = (
            [SEG_PRIOR_TEXT] * len(prior_ids)
            + [SEG_UPCOMING_TEXT] * len(upcoming_ids)
        )

        text_ids_t = torch.tensor(all_ids, dtype=torch.long)
        text_segs_t = torch.tensor(all_segs, dtype=torch.long)

        return context, target, text_ids_t, text_segs_t


# ═══════════════════════════════════════════════════════════════════
# 2. Collate
# ═══════════════════════════════════════════════════════════════════
def collate_fn(batch):
    """Pad text_ids and text_segments to max length in batch."""
    contexts, targets, text_ids_list, text_segs_list = zip(*batch)

    contexts = torch.stack(contexts)    # [B, 375, 128]
    targets = torch.stack(targets)      # [B, 75, 128]

    max_text_len = max(len(t) for t in text_ids_list)
    if max_text_len == 0:
        max_text_len = 1

    B = len(batch)
    text_ids = torch.full((B, max_text_len), CHAR_PAD_ID, dtype=torch.long)
    text_segs = torch.zeros((B, max_text_len), dtype=torch.long)

    for i, (ids, segs) in enumerate(zip(text_ids_list, text_segs_list)):
        if len(ids) > 0:
            text_ids[i, : len(ids)] = ids
            text_segs[i, : len(segs)] = segs

    return contexts, targets, text_ids, text_segs


# ═══════════════════════════════════════════════════════════════════
# 3. Sampling
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_samples(model, dataset, epoch, n=10):
    """Euler-sample n clips conditioned on context + text, decode to audio."""
    model.eval()
    out_dir = SAMPLE_DIR / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = Codec()
    indices = torch.randperm(len(dataset))[:n].tolist()

    for j, idx in enumerate(indices):
        context, _target, text_ids, text_segs = dataset[idx]
        context = context.unsqueeze(0).to(DEVICE)
        text_ids = text_ids.unsqueeze(0).to(DEVICE)
        text_segs = text_segs.unsqueeze(0).to(DEVICE)

        z = torch.randn(1, TARGET_FRAMES, 128, device=DEVICE)
        dt = 1.0 / N_EULER

        for i in range(N_EULER):
            t_i = torch.tensor([i / N_EULER], device=DEVICE)
            v = model(z, t_i, context, text_ids, text_segs)
            z = z + v * dt

        # Save predicted 1 s
        codec.decode_continuous(
            z.permute(0, 2, 1).cpu(),
            str(out_dir / f"pred_{j:02d}.mp3"),
        )
        # Save full clip (5 s context + 1 s generated)
        full = torch.cat([context.cpu(), z.cpu()], dim=1)
        codec.decode_continuous(
            full.permute(0, 2, 1),
            str(out_dir / f"full_{j:02d}.mp3"),
        )

    print(f"  Saved {n} samples → {out_dir}")
    model.train()


# ═══════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════
def main():
    # --- dataset ---
    dataset = LibriTTSFlowDataset()
    if len(dataset) == 0:
        print("No usable utterances. Check preprocess.py output.", file=sys.stderr)
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    # --- model ---
    model = RectifiedFlowPerceiver(
        input_dim=128,
        dim=256,
        num_heads=8,
        num_latent_layers=12,
        max_latents=256,
        max_seq_len=2048,
    ).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- training ---
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (context, target, text_ids, text_segs) in enumerate(loader):
            context = context.to(DEVICE)
            target = target.to(DEVICE)
            text_ids = text_ids.to(DEVICE)
            text_segs = text_segs.to(DEVICE)
            B = target.shape[0]

            t = torch.rand(B, device=DEVICE)
            x0 = torch.randn_like(target)

            t_broad = t.view(B, 1, 1)
            z_t = (1.0 - t_broad) * x0 + t_broad * target
            v_target = target - x0

            v_pred = model(z_t, t, context, text_ids, text_segs)

            loss = (v_pred - v_target).abs().mean()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % PRINT_EVERY == 0:
                print(
                    f"  epoch {epoch} | batch {batch_idx + 1}/{len(loader)}"
                    f" | loss = {loss.item():.6f}"
                )

        avg = epoch_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch}/{EPOCHS} — avg loss = {avg:.6f}"
            f" | {len(dataset)} samples"
        )

        generate_samples(model, dataset, epoch, n=10)

    print("Training complete.")


if __name__ == "__main__":
    main()
