"""
Online data processing pipeline for peoples_speech MP3s.

Architecture (single asyncio event loop):
  1. process_batch(batch_size=16) picks random MP3s from data/
  2. For each: cut a random 5-second chunk, then in parallel:
     a) transcribe via Gemini (async, natively in the event loop)
     b) compute per-100ms RMS loudness (CPU-bound, run_in_executor)
  3. Normalize transcription to 128 chars
  4. Batch-encode all chunks through EnCodec → continuous embeddings
  5. Return list of {transcription, loudness, encodec_emb}
"""

import asyncio
import os
import random
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from pydantic import BaseModel

# ── Transcription setup (Gemini) — LAZY init ────────────────────────
_key_path = Path.home() / ".agents" / "secrets" / "google_ai_studio"
if _key_path.exists():
    os.environ.setdefault("GOOGLE_API_KEY", _key_path.read_text().strip())
    print("[dataProc] Loaded GOOGLE_API_KEY from secrets file")
else:
    print("[dataProc] WARNING: no secrets file at", _key_path)

from pydantic_ai import Agent, BinaryContent  # noqa: E402
from pydantic_ai.models.google import GoogleModel  # noqa: E402
from pydantic_ai.providers.google import GoogleProvider  # noqa: E402

# Agent is created LAZILY to avoid binding asyncio primitives to the
# wrong event loop (module import happens before asyncio.run()).
_agent = None


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        print("[dataProc] Creating Gemini agent (lazy init inside running loop)...")
        provider = GoogleProvider()
        model = GoogleModel("gemini-3.1-flash-lite", provider=provider)
        _agent = Agent(model)
        print("[dataProc] Gemini agent created OK")
    return _agent


# ── Constants ───────────────────────────────────────────────────────
DATA_DIR = Path("data")
CHUNK_SECONDS = 5.0
TARGET_SR = 24_000
NUM_CHARS = 128
LOUDNESS_BLOCK_MS = 100

_executor = ProcessPoolExecutor(max_workers=4)


# ── Data models ─────────────────────────────────────────────────────

class TranscribeOut(BaseModel):
    transcription: str
    is_empty: bool


# ── Codec ───────────────────────────────────────────────────────────

class Codec:
    """Wraps EnCodec 24 kHz for continuous encode/decode.  Uses GPU when available."""

    def __init__(self, bandwidth: float = 6.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Codec] Loading EnCodec 24kHz on {self.device}...")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(self.device)
        self.sample_rate = self.model.sample_rate  # 24000
        self.channels = self.model.channels        # 1
        print(f"[Codec] Ready (sr={self.sample_rate}, bw={bandwidth})")

    def encode_continuous_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """[1, T_samples] mono waveform at 24 kHz → [1, 128, T_frames] (returned on CPU)."""
        wav = wav.unsqueeze(0).to(self.device)  # [1, 1, T]
        with torch.no_grad():
            emb = self.model.encoder(wav)
            codes = self.model.quantizer.encode(
                emb, self.model.frame_rate, self.model.bandwidth
            )
            emb_q = self.model.quantizer.decode(codes)
        return emb_q.cpu()

    def decode_continuous(self, emb: torch.Tensor, output_path: str) -> None:
        """[1, 128, T] → audio file."""
        with torch.no_grad():
            audio = self.model.decoder(emb.to(self.device))
        audio = audio.squeeze(0)
        torchaudio.save(output_path, audio.cpu(), self.sample_rate)


# Module-level singleton — loaded once, lives on GPU if available.
_codec = Codec()


# ── Transcription ───────────────────────────────────────────────────

async def transcribe(mp3_bytes: bytes) -> TranscribeOut:
    """Async Gemini transcription — runs natively in the event loop."""
    agent = _get_agent()
    print(f"[transcribe] Sending {len(mp3_bytes)} bytes to Gemini...")
    try:
        result = await agent.run(
            [
                "Transcribe this audio exactly. "
                "Return ONLY the spoken words, nothing else.",
                BinaryContent(data=mp3_bytes, media_type="audio/mpeg"),
            ]
        )
        text = result.output or ""
        clean = "".join(
            ch for ch in text.lower() if ch.isalnum() or ch == " "
        ).strip()
        is_empty = len(clean) == 0
        print(f"[transcribe] Got {len(clean)} chars, empty={is_empty}: '{clean[:60]}...'")
        return TranscribeOut(transcription=clean, is_empty=is_empty)
    except Exception as e:
        print(f"[transcribe] ERROR: {e}")
        traceback.print_exc()
        return TranscribeOut(transcription="", is_empty=True)


# ── Loudness ────────────────────────────────────────────────────────

def _compute_loudness(wav_mono: np.ndarray, sr: int) -> np.ndarray:
    """Per-100 ms block RMS amplitude.  Returns 1-D float32 ndarray."""
    block_samples = int(sr * LOUDNESS_BLOCK_MS / 1000)
    n_blocks = len(wav_mono) // block_samples
    if n_blocks == 0:
        return np.array([], dtype=np.float32)
    trimmed = wav_mono[: n_blocks * block_samples]
    blocks = trimmed.reshape(n_blocks, block_samples)
    return np.sqrt(np.mean(blocks ** 2, axis=1)).astype(np.float32)


async def get_loudness(wav_mono: np.ndarray, sr: int) -> np.ndarray:
    """Async CPU-bound loudness via process pool executor."""
    print(f"[loudness] Computing RMS for {len(wav_mono)} samples...")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _compute_loudness, wav_mono, sr)
    print(f"[loudness] Done — {len(result)} blocks")
    return result


# ── Transcription normalisation ─────────────────────────────────────

_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 ")


def normalize_transcription(text: str, length: int = NUM_CHARS) -> str:
    """
    Lowercase, keep a-z / 0-9 / space, replace anything else with '?'.
    Pad or truncate to exactly ``length`` characters.
    """
    out = [ch if ch in _ALLOWED else "?" for ch in text.lower()]
    s = "".join(out)
    if len(s) >= length:
        return s[:length]
    return s + "?" * (length - len(s))


# ── Preprocess a single clip ────────────────────────────────────────

async def preprocess(mp3_path: str) -> dict | None:
    """
    Load MP3, cut a random 5 s chunk, run transcribe + loudness in
    parallel.  Returns None if the clip is too short or silent.
    """
    print(f"[preprocess] Loading {mp3_path}...")
    try:
        wav, sr = torchaudio.load(mp3_path)
    except Exception as e:
        print(f"[preprocess] FAILED to load {mp3_path}: {e}")
        return None

    # Resample + mono
    if sr != TARGET_SR:
        print(f"[preprocess] Resampling {sr} → {TARGET_SR}")
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    total_samples = wav.shape[1]
    chunk_samples = int(CHUNK_SECONDS * sr)

    if total_samples < chunk_samples:
        print(f"[preprocess] Too short ({total_samples} < {chunk_samples}), skipping")
        return None

    # Random 5-second window
    start = random.randint(0, total_samples - chunk_samples)
    chunk_wav = wav[:, start : start + chunk_samples]  # [1, chunk_samples]
    print(f"[preprocess] Cut 5s chunk at sample {start} from {Path(mp3_path).name}")

    # MP3 bytes for Gemini
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        torchaudio.save(tmp.name, chunk_wav, sr, format="mp3")
        mp3_bytes = Path(tmp.name).read_bytes()
    print(f"[preprocess] Encoded chunk to {len(mp3_bytes)} MP3 bytes")

    wav_np = chunk_wav.squeeze(0).numpy()

    # Transcription (I/O) + loudness (CPU) in parallel
    print(f"[preprocess] Launching transcribe + loudness in parallel...")
    try:
        transcribe_out, loudness = await asyncio.gather(
            transcribe(mp3_bytes),
            get_loudness(wav_np, sr),
        )
    except Exception as e:
        print(f"[preprocess] gather FAILED: {e}")
        traceback.print_exc()
        return None

    if transcribe_out.is_empty:
        print(f"[preprocess] Empty transcription, skipping")
        return None

    norm = normalize_transcription(transcribe_out.transcription)
    print(f"[preprocess] OK — transcript='{norm[:40]}...' loudness={len(loudness)} blocks")

    return {
        "transcription": norm,
        "loudness": loudness,
        "wav_chunk": chunk_wav,  # [1, chunk_samples] @ 24 kHz
        "mp3_path": mp3_path,
    }


# ── Batch encode through EnCodec ────────────────────────────────────

def _encode_batch_sync(wav_chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Encode list of [1, T] waveforms using the module-level GPU codec singleton."""
    results = []
    for i, w in enumerate(wav_chunks):
        print(f"[encodec] Encoding chunk {i+1}/{len(wav_chunks)} shape={w.shape}...")
        results.append(_codec.encode_continuous_waveform(w))
    print(f"[encodec] Batch done — {len(results)} items encoded")
    return results


async def encode_batch(preprocessed: list[dict]) -> list[dict]:
    """
    Take preprocessed items, batch-encode wav_chunks through EnCodec
    on GPU (single thread via asyncio.to_thread).
    """
    wav_chunks = [item["wav_chunk"] for item in preprocessed]
    print(f"[encode_batch] Encoding {len(wav_chunks)} chunks via EnCodec...")

    embeddings = await asyncio.to_thread(_encode_batch_sync, wav_chunks)

    print(f"[encode_batch] All {len(embeddings)} chunks encoded")
    return [
        {
            "transcription": item["transcription"],
            "loudness": item["loudness"],
            "encodec_emb": emb,  # [1, 128, T_frames]
        }
        for item, emb in zip(preprocessed, embeddings)
    ]


# ── Global process operation ────────────────────────────────────────

async def process_batch(batch_size: int = 16) -> list[dict]:
    """
    Pick ``batch_size`` random MP3s from data/, preprocess in parallel
    (transcribe + loudness), then batch-encode through EnCodec.
    """
    mp3s = list(DATA_DIR.glob("*.mp3"))
    if not mp3s:
        raise FileNotFoundError(f"No MP3 files found in {DATA_DIR}")

    t0 = time.perf_counter()

    chosen = random.choices(mp3s, k=batch_size)
    print(f"[process_batch] Picked {len(chosen)} random MP3s from {len(mp3s)} available")

    # Parallel preprocess (asyncio tasks)
    print(f"[process_batch] Launching {len(chosen)} preprocess tasks...")
    t_preproc = time.perf_counter()
    results = await asyncio.gather(
        *(preprocess(str(p)) for p in chosen)
    )
    dt_preproc = time.perf_counter() - t_preproc

    valid = [r for r in results if r is not None]
    n_failed = len(results) - len(valid)
    print(f"[process_batch] Preprocess done in {dt_preproc:.2f}s: {len(valid)} valid, {n_failed} skipped/failed")

    if not valid:
        print("[process_batch] WARNING: no valid items in batch!")
        return []

    # Batch EnCodec encoding
    t_enc = time.perf_counter()
    result = await encode_batch(valid)
    dt_enc = time.perf_counter() - t_enc

    dt_total = time.perf_counter() - t0
    print(f"[process_batch] Batch complete — {len(result)} items ready "
          f"(preproc={dt_preproc:.2f}s, encodec={dt_enc:.2f}s, total={dt_total:.2f}s)")
    return result
