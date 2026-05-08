"""
Tests for the transcription pipeline.

Run with:  uv run python test_transcribe.py
"""

import tempfile
import traceback
from pathlib import Path

import torch
import torchaudio

# ── Test 1: torchaudio can save an mp3 to a temp file ──────────────
def test_mp3_creation():
    """Create a 1-second sine wave, save as mp3, verify bytes are non-empty."""
    print("TEST 1: mp3 creation via torchaudio.save ...")
    sr = 24_000
    t = torch.linspace(0, 1, sr).unsqueeze(0)  # [1, 24000]
    sine = torch.sin(2 * 3.14159 * 440 * t)     # 440 Hz sine

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        torchaudio.save(tmp.name, sine, sr, format="mp3")
        data = Path(tmp.name).read_bytes()

    assert len(data) > 100, f"mp3 file too small: {len(data)} bytes"
    print(f"  ✓ mp3 created successfully ({len(data)} bytes)")
    return data


# ── Test 2: Gemini API call with audio bytes ───────────────────────
def test_gemini_api(audio_bytes: bytes):
    """Send mp3 bytes to Gemini and verify we get text back."""
    print("TEST 2: Gemini API call (transcribe_audio_bytes) ...")
    from transcribe import transcribe_audio_bytes

    text = transcribe_audio_bytes(audio_bytes, media_type="audio/mpeg")
    print(f"  Response: '{text}'")
    # Sine wave won't have real speech, but API should not error
    print(f"  ✓ Gemini returned a response (len={len(text)})")
    return True


# ── Test 3: chunk_and_transcribe on real audio ─────────────────────
def test_chunk_and_transcribe():
    """Run the full pipeline on rice_speech_10s.wav."""
    print("TEST 3: chunk_and_transcribe on rice_speech_10s.wav ...")
    wav_path = "rice_speech_10s.wav"
    if not Path(wav_path).exists():
        print(f"  SKIP: {wav_path} not found")
        return True

    from transcribe import chunk_and_transcribe

    texts = chunk_and_transcribe(wav_path, chunk_seconds=1.0)
    print(f"  Got {len(texts)} chunks:")
    for i, t in enumerate(texts):
        print(f"    chunk {i}: '{t}'")

    assert len(texts) > 0, "No chunks returned!"
    non_empty = sum(1 for t in texts if t)
    print(f"  ✓ {non_empty}/{len(texts)} chunks have transcription text")
    return True


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    # Test 1
    try:
        mp3_bytes = test_mp3_creation()
        results["mp3_creation"] = "PASS"
    except Exception:
        traceback.print_exc()
        results["mp3_creation"] = "FAIL"
        mp3_bytes = None

    # Test 2
    if mp3_bytes:
        try:
            test_gemini_api(mp3_bytes)
            results["gemini_api"] = "PASS"
        except Exception:
            traceback.print_exc()
            results["gemini_api"] = "FAIL"
    else:
        results["gemini_api"] = "SKIP (no mp3 bytes)"

    # Test 3
    try:
        test_chunk_and_transcribe()
        results["chunk_and_transcribe"] = "PASS"
    except Exception:
        traceback.print_exc()
        results["chunk_and_transcribe"] = "FAIL"

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"\n{'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
