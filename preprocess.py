"""
Pre-encode LibriTTS-R train-clean-100 through EnCodec.

Walks all .wav files, encodes each into continuous embeddings [T, 128],
saves as .pt, and writes a manifest.json for the training script.

Uses multiprocessing to parallelise the CPU-bound EnCodec work.

Usage:
    uv run python preprocess.py
"""

import json
import subprocess
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch

from codec import Codec

# ── paths ────────────────────────────────────────────────────────────
DATA_DIR = Path("data/libritts_r")
CACHE_DIR = Path("data/encodec_cache")
MANIFEST_PATH = CACHE_DIR / "manifest.json"

FPS = 75
CHUNK_FRAMES = 1 * FPS  # 75 frames per 1-second chunk


# ═════════════════════════════════════════════════════════════════════
# 1. Download & extract (same as before)
# ═════════════════════════════════════════════════════════════════════
def download_and_extract():
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


# ═════════════════════════════════════════════════════════════════════
# 2. Encode a single file (called by worker processes)
# ═════════════════════════════════════════════════════════════════════
def encode_one(wav_path: str) -> dict | None:
    """Encode a single wav → .pt.  Returns manifest entry or None on error."""
    stem = Path(wav_path).stem
    cache_path = CACHE_DIR / f"{stem}.pt"

    # Skip if already encoded
    if cache_path.exists():
        try:
            emb = torch.load(cache_path, weights_only=True)
            n_frames = emb.shape[0]
            n_chunks = n_frames // CHUNK_FRAMES
            return {
                "stem": stem,
                "pt_path": str(cache_path),
                "wav_path": wav_path,
                "n_frames": n_frames,
                "n_chunks": n_chunks,
            }
        except Exception:
            cache_path.unlink(missing_ok=True)

    try:
        codec = Codec()
        emb = codec.encode_continuous(wav_path)   # [1, 128, T]
        emb = emb.squeeze(0).permute(1, 0)        # [T, 128]
        torch.save(emb, cache_path)
    except Exception as e:
        print(f"  [encode] skip {Path(wav_path).name}: {e}")
        return None

    n_frames = emb.shape[0]
    n_chunks = n_frames // CHUNK_FRAMES
    return {
        "stem": stem,
        "pt_path": str(cache_path),
        "wav_path": wav_path,
        "n_frames": n_frames,
        "n_chunks": n_chunks,
    }


# ═════════════════════════════════════════════════════════════════════
# 3. Main
# ═════════════════════════════════════════════════════════════════════
def main():
    download_and_extract()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(str(p) for p in DATA_DIR.rglob("*.wav"))
    print(f"Found {len(wav_paths)} wav files.")

    # Load existing manifest so we can skip already-encoded files
    existing: dict[str, dict] = {}
    if MANIFEST_PATH.exists():
        try:
            existing = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}

    # Filter to only un-encoded files
    todo = [p for p in wav_paths if Path(p).stem not in existing]
    print(f"{len(existing)} already encoded, {len(todo)} remaining.")

    if todo:
        n_workers = min(cpu_count(), 8)
        print(f"Encoding with {n_workers} workers …")

        manifest = dict(existing)
        done = 0

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(encode_one, todo):
                if result is not None:
                    manifest[result["stem"]] = result
                done += 1
                if done % 200 == 0:
                    # Periodic save for resumability
                    MANIFEST_PATH.write_text(json.dumps(manifest))
                    print(f"  {done}/{len(todo)} encoded …")

        # Final save
        MANIFEST_PATH.write_text(json.dumps(manifest))
        print(f"Encoding complete. {len(manifest)} entries in manifest.")
    else:
        print("All files already encoded.")


if __name__ == "__main__":
    main()

