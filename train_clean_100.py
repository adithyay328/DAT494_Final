"""
Text-conditioned rectified-flow training on LibriTTS-R train-clean-100.

- Background thread encodes all utterances through EnCodec + transcribes
  each 1-second chunk via Gemini 2.5 Flash Lite (perfect text alignment)
- Foreground trains rectified flow: 5 s context + text → predict next 1 s
- Every epoch: Euler-sample 10 clips, decode, and save as audio

Usage:
    uv run python train_clean_100.py
"""

import json
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from codec import Codec
from perceiver import (
    CHAR_PAD_ID,
    SEG_PRIOR_TEXT,
    SEG_UPCOMING_TEXT,
    RectifiedFlowPerceiver,
    text_to_ids,
)
from transcribe import chunk_and_transcribe

# ── Constants ───────────────────────────────────────────────────────
FPS = 75  # EnCodec frame rate
CHUNK_FRAMES = 1 * FPS      # 75   (1-second chunk)
CTX_CHUNKS = 5               # 5 chunks = 5 s context
TGT_CHUNKS = 1               # 1 chunk  = 1 s target
MIN_CHUNKS = CTX_CHUNKS + TGT_CHUNKS  # 6 chunks = 6 s minimum

CONTEXT_FRAMES = CTX_CHUNKS * CHUNK_FRAMES   # 375
TARGET_FRAMES = TGT_CHUNKS * CHUNK_FRAMES    # 75

DATA_DIR = Path("data/libritts_r")
CACHE_DIR = Path("data/encodec_cache")
SAMPLE_DIR = Path("data/samples")
MANIFEST_PATH = CACHE_DIR / "manifest.json"

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
N_EULER = 100
PRINT_EVERY = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════
# 1. Download & extract
# ═══════════════════════════════════════════════════════════════════
def download_and_extract():
    """Download LibriTTS-R train-clean-100 from openslr if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(DATA_DIR.rglob("*.wav")):
        print("Dataset already extracted.")
        return

    tarball = DATA_DIR / "train_clean_100.tar.gz"
    if not tarball.exists() or tarball.stat().st_size < 1_000_000:
        if tarball.exists():
            tarball.unlink()
        url = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
        print(f"Downloading {url} (~8.1 GB) …")
        subprocess.run(
            ["wget", "--no-check-certificate", "-q", "--show-progress",
             "-O", str(tarball), url],
            check=True,
        )
        print("Download complete.")

    print("Extracting (may take a few minutes) …")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Extraction complete.")


def find_wav_files() -> list[str]:
    """Return all .wav paths under DATA_DIR."""
    wavs = sorted(str(p) for p in DATA_DIR.rglob("*.wav"))
    print(f"Found {len(wavs)} wav files.")
    return wavs


# ═══════════════════════════════════════════════════════════════════
# 2. Background encoding + Gemini transcription
# ═══════════════════════════════════════════════════════════════════
def encode_background(wav_paths: list[str], done_event: threading.Event):
    """
    For each utterance:
      1. Encode through EnCodec → save [T, 128] tensor
      2. Split raw audio into 1-s chunks, transcribe each via Gemini
      3. Store per-chunk texts in manifest
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    codec = Codec()

    manifest: dict = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            manifest = {}

    for i, wav_path in enumerate(wav_paths):
        stem = Path(wav_path).stem
        if stem in manifest:
            continue

        cache_path = CACHE_DIR / f"{stem}.pt"

        # ① EnCodec encode (must succeed)
        try:
            emb = codec.encode_continuous(wav_path)  # [1, 128, T]
            emb = emb.squeeze(0).permute(1, 0)       # [T, 128]
            torch.save(emb, cache_path)
        except Exception as e:
            print(f"  [encode] skip {Path(wav_path).name}: {e}")
            continue

        n_frames = emb.shape[0]
        n_full_chunks = n_frames // CHUNK_FRAMES

        # ② Gemini per-chunk transcription (best-effort, empty on failure)
        try:
            chunk_texts = chunk_and_transcribe(wav_path, chunk_seconds=1.0)
            chunk_texts = chunk_texts[:n_full_chunks]
            while len(chunk_texts) < n_full_chunks:
                chunk_texts.append("")
        except Exception as e:
            print(f"  [transcribe] fail {Path(wav_path).name}: {e}")
            chunk_texts = [""] * n_full_chunks

        manifest[stem] = {
            "path": str(cache_path),
            "n_frames": n_frames,
            "n_chunks": n_full_chunks,
            "chunk_texts": chunk_texts,
        }

        # Save manifest after EVERY file so training can start ASAP
        MANIFEST_PATH.write_text(json.dumps(manifest))
        if (i + 1) % 100 == 0:
            print(f"  [encode] {i + 1}/{len(wav_paths)} encoded+transcribed")

    MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"  [encode] Finished all {len(wav_paths)} files.")
    done_event.set()


# ═══════════════════════════════════════════════════════════════════
# 3. Dataset (chunk-aligned)
# ═══════════════════════════════════════════════════════════════════
class LibriTTSFlowDataset(Dataset):
    """
    Returns (context_audio, target_audio, text_ids, text_segments).

    Each training sample picks 6 consecutive 1-s chunks from one utterance:
      - chunks [i..i+4] → context audio (375 frames) + prior text
      - chunk  [i+5]    → target audio  (75 frames)  + upcoming text

    Text alignment is exact: each chunk was transcribed independently.
    """

    def __init__(self):
        if not MANIFEST_PATH.exists():
            self.entries: list[tuple[str, int, list[str]]] = []
            return
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            self.entries = []
            return
        self.entries = [
            (v["path"], v["n_chunks"], v["chunk_texts"])
            for v in manifest.values()
            if v.get("n_chunks", 0) >= MIN_CHUNKS
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, n_chunks, chunk_texts = self.entries[idx]
        emb = torch.load(path, weights_only=True)  # [T, 128]

        # Pick random start chunk (chunk-aligned)
        max_start = n_chunks - MIN_CHUNKS
        start = torch.randint(0, max(1, max_start + 1), (1,)).item()

        # Context: 5 chunks
        ctx_s = start * CHUNK_FRAMES
        ctx_e = (start + CTX_CHUNKS) * CHUNK_FRAMES
        context = emb[ctx_s:ctx_e]  # [375, 128]

        # Target: 1 chunk
        tgt_s = (start + CTX_CHUNKS) * CHUNK_FRAMES
        tgt_e = (start + CTX_CHUNKS + TGT_CHUNKS) * CHUNK_FRAMES
        target = emb[tgt_s:tgt_e]  # [75, 128]

        # Text: prior = context chunks, upcoming = target chunk
        prior_text = " ".join(chunk_texts[start : start + CTX_CHUNKS])
        upcoming_text = chunk_texts[start + CTX_CHUNKS]

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
# 4. Sampling
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
# 5. Main
# ═══════════════════════════════════════════════════════════════════
def main():
    download_and_extract()
    wav_paths = find_wav_files()

    # --- background encoding + transcription ---
    done_event = threading.Event()
    enc_thread = threading.Thread(
        target=encode_background,
        args=(wav_paths, done_event),
        daemon=True,
    )
    enc_thread.start()

    # Wait for enough usable manifest entries (≥ MIN_CHUNKS each)
    MIN_USABLE = 10
    print(f"Waiting for ≥{MIN_USABLE} usable manifest entries …")
    while True:
        n_usable = 0
        if MANIFEST_PATH.exists():
            try:
                m = json.loads(MANIFEST_PATH.read_text())
                n_usable = sum(
                    1 for v in m.values()
                    if v.get("n_chunks", 0) >= MIN_CHUNKS
                )
            except (json.JSONDecodeError, Exception):
                pass
        if n_usable >= MIN_USABLE:
            break
        print(f"  {n_usable} usable entries …")
        time.sleep(5)
    print(f"Starting training ({n_usable} usable entries ready).")

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
        dataset = LibriTTSFlowDataset()
        if len(dataset) == 0:
            print(f"Epoch {epoch}: no usable data yet, waiting …")
            time.sleep(10)
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
        )

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
        status = "✓" if done_event.is_set() else "encoding…"
        print(
            f"Epoch {epoch}/{EPOCHS} — avg loss = {avg:.6f}"
            f" | {len(dataset)} samples [{status}]"
        )

        generate_samples(model, dataset, epoch, n=10)

    print("Training complete.")


if __name__ == "__main__":
    main()
