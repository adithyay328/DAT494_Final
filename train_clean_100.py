"""
Rectified-flow training on LibriTTS-R train-clean-100.

- Background thread encodes all utterances through EnCodec → .pt cache
- Foreground trains rectified flow: 5 s context → predict next 10 s
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
from perceiver import RectifiedFlowPerceiver

# ── Constants ───────────────────────────────────────────────────────
FPS = 75  # EnCodec frame rate
CONTEXT_FRAMES = 5 * FPS  # 375  (5 seconds)
TARGET_FRAMES = 10 * FPS  # 750  (10 seconds)
MIN_FRAMES = TARGET_FRAMES  # utterance must be ≥ 10 s

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

    # Already extracted?
    if any(DATA_DIR.rglob("*.wav")):
        print("Dataset already extracted.")
        return

    tarball = DATA_DIR / "train_clean_100.tar.gz"
    # Re-download if missing or suspiciously small (failed prior download)
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
    """Return all .wav paths under DATA_DIR (any nesting level)."""
    wavs = sorted(str(p) for p in DATA_DIR.rglob("*.wav"))
    print(f"Found {len(wavs)} wav files.")
    return wavs


# ═══════════════════════════════════════════════════════════════════
# 2. Background encoding
# ═══════════════════════════════════════════════════════════════════
def encode_background(wav_paths: list[str], done_event: threading.Event):
    """Encode all wavs via EnCodec, save .pt files + manifest."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    codec = Codec()  # separate instance for thread safety

    # Resume from existing manifest
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
        try:
            emb = codec.encode_continuous(wav_path)  # [1, 128, T]
            emb = emb.squeeze(0).permute(1, 0)  # [T, 128]
            torch.save(emb, cache_path)
            manifest[stem] = {
                "path": str(cache_path),
                "n_frames": emb.shape[0],
            }
        except Exception as e:
            print(f"  [encode] skip {Path(wav_path).name}: {e}")
            continue

        if (i + 1) % 100 == 0:
            MANIFEST_PATH.write_text(json.dumps(manifest))
        if (i + 1) % 500 == 0:
            print(f"  [encode] {i + 1}/{len(wav_paths)} encoded")

    MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"  [encode] Finished all {len(wav_paths)} files.")
    done_event.set()


# ═══════════════════════════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════════════════════════
class LibriTTSFlowDataset(Dataset):
    """
    (context, target) pairs from cached EnCodec embeddings.

    context : [CONTEXT_FRAMES, 128]  — 5 s clean (zero-padded left if short)
    target  : [TARGET_FRAMES,  128]  — 10 s clean
    """

    def __init__(self):
        if not MANIFEST_PATH.exists():
            self.entries: list[tuple[str, int]] = []
            return
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            self.entries = []
            return
        self.entries = [
            (v["path"], v["n_frames"])
            for v in manifest.values()
            if v["n_frames"] >= MIN_FRAMES
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, n_frames = self.entries[idx]
        emb = torch.load(path, weights_only=True)  # [T, 128]
        T = emb.shape[0]

        avail_ctx = T - TARGET_FRAMES
        ctx_len = min(max(avail_ctx, 0), CONTEXT_FRAMES)

        max_start = T - ctx_len - TARGET_FRAMES
        start = torch.randint(0, max(1, max_start + 1), (1,)).item()

        context = emb[start : start + ctx_len]
        target = emb[start + ctx_len : start + ctx_len + TARGET_FRAMES]

        # Left-pad context to fixed CONTEXT_FRAMES with zeros
        if ctx_len < CONTEXT_FRAMES:
            pad = torch.zeros(CONTEXT_FRAMES - ctx_len, 128)
            context = torch.cat([pad, context], dim=0)

        return context, target  # [375, 128], [750, 128]


# ═══════════════════════════════════════════════════════════════════
# 4. Sampling
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_samples(model, dataset, epoch, n=10):
    """Euler-sample n clips conditioned on context, decode to audio."""
    model.eval()
    out_dir = SAMPLE_DIR / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = Codec()
    indices = torch.randperm(len(dataset))[:n].tolist()

    for j, idx in enumerate(indices):
        context, _target = dataset[idx]
        context = context.unsqueeze(0).to(DEVICE)  # [1, 375, 128]

        z = torch.randn(1, TARGET_FRAMES, 128, device=DEVICE)
        dt = 1.0 / N_EULER

        for i in range(N_EULER):
            t_i = torch.tensor([i / N_EULER], device=DEVICE)
            inp = torch.cat([context, z], dim=1)  # [1, 1125, 128]
            v = model(inp, t_i)[:, CONTEXT_FRAMES:, :]  # [1, 750, 128]
            z = z + v * dt

        # Save predicted 10 s
        codec.decode_continuous(
            z.permute(0, 2, 1).cpu(),
            str(out_dir / f"pred_{j:02d}.mp3"),
        )
        # Save full clip (5 s context + 10 s generated)
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

    # --- background encoding ---
    done_event = threading.Event()
    enc_thread = threading.Thread(
        target=encode_background,
        args=(wav_paths, done_event),
        daemon=True,
    )
    enc_thread.start()

    # wait for a minimum pool of cached files
    print("Waiting for initial encoding …")
    while True:
        n_cached = len(list(CACHE_DIR.glob("*.pt"))) if CACHE_DIR.exists() else 0
        if n_cached >= 200:
            break
        print(f"  {n_cached} files cached …")
        time.sleep(5)
    print(f"Starting training ({n_cached} files ready).")

    # --- model ---
    model = RectifiedFlowPerceiver(
        input_dim=128,
        dim=256,
        num_heads=8,
        num_latent_layers=6,
        max_latents=128,
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
        )

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (context, target) in enumerate(loader):
            context = context.to(DEVICE)  # [B, 375, 128]
            target = target.to(DEVICE)  # [B, 750, 128]
            B = target.shape[0]

            t = torch.rand(B, device=DEVICE)
            x0 = torch.randn_like(target)

            t_broad = t.view(B, 1, 1)
            z_t = (1.0 - t_broad) * x0 + t_broad * target
            v_target = target - x0

            # clean context ++ noisy target
            inp = torch.cat([context, z_t], dim=1)  # [B, 1125, 128]
            v_pred = model(inp, t)[:, CONTEXT_FRAMES:, :]  # [B, 750, 128]

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
