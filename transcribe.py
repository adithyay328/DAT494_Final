"""
Per-chunk audio transcription via PydanticAI + Gemini 2.5 Flash Lite.

Each 1-second audio chunk is sent independently to Gemini, so the returned
text is perfectly aligned to the chunk — no forced alignment needed.

API key is read from ~/.agents/secrets/google_ai_studio at import time.
"""

import os
import tempfile
import time
from pathlib import Path

import torch
import torchaudio

# ── Load API key from secrets (never printed / logged) ──────────────
_key_path = Path.home() / ".agents" / "secrets" / "google_ai_studio"
if _key_path.exists():
    os.environ.setdefault("GOOGLE_API_KEY", _key_path.read_text().strip())

from pydantic_ai import Agent, BinaryContent  # noqa: E402
from pydantic_ai.models.google import GoogleModel  # noqa: E402
from pydantic_ai.providers.google import GoogleProvider  # noqa: E402

_provider = GoogleProvider()
_model = GoogleModel("gemini-3.1-flash-lite", provider=_provider)
_agent = Agent(_model)

# Rate-limit: small sleep between API calls to stay under quota
_MIN_DELAY = 0.05  # 50 ms → ~20 req/s


def transcribe_audio_bytes(audio_bytes: bytes, media_type: str = "audio/mpeg") -> str:
    """Send audio bytes to Gemini and return cleaned transcription."""
    result = _agent.run_sync(
        [
            "Transcribe this audio exactly. "
            "Return ONLY the spoken words, nothing else.",
            BinaryContent(data=audio_bytes, media_type=media_type),
        ]
    )
    text = result.output or ""
    # Clean: lowercase, keep alphanumeric + space only
    clean = "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()
    return clean


def chunk_and_transcribe(
    wav_path: str,
    chunk_seconds: float = 1.0,
    target_sr: int = 24_000,
) -> list[str]:
    """
    Load a wav, split into fixed-length chunks, transcribe each via Gemini.

    Parameters
    ----------
    wav_path : str
        Path to the audio file (any format torchaudio supports).
    chunk_seconds : float
        Duration of each chunk in seconds (default 1.0).
    target_sr : int
        Sample rate to resample to before chunking (default 24 kHz).

    Returns
    -------
    list[str]
        One cleaned transcription string per full chunk.
        Partial trailing chunks (< 0.5 s) are skipped.
    """
    wav, sr = torchaudio.load(wav_path)

    # Resample + mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    chunk_samples = int(chunk_seconds * target_sr)
    total_samples = wav.shape[1]

    texts: list[str] = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, start:end]

        # Skip very short trailing chunks (< 0.5 s)
        if chunk.shape[1] < target_sr * 0.5:
            break

        # Save chunk to a temp mp3 file, read bytes, send to Gemini
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                torchaudio.save(tmp.name, chunk, target_sr, format="mp3")
                audio_bytes = Path(tmp.name).read_bytes()
            text = transcribe_audio_bytes(audio_bytes, media_type="audio/mpeg")
        except Exception as e:
            print(f"  [transcribe] chunk {len(texts)} error: {e}")
            text = ""

        texts.append(text)
        time.sleep(_MIN_DELAY)

    return texts
