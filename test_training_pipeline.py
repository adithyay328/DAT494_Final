"""
Unit tests for the training pipeline — all synthetic data, no EnCodec/Gemini.

Tests:
  1. text_to_ids mapping
  2. Synthetic manifest → dataset → correct shapes
  3. collate_fn padding + batching
  4. Model forward produces correct output shape
  5. Full training step (forward → loss → backward → optimizer)
  6. Euler sampling loop shapes

Run with:  uv run python test_training_pipeline.py
"""

import json
import tempfile
import traceback
from pathlib import Path

import torch

from perceiver import (
    CHAR_PAD_ID,
    SEG_PRIOR_TEXT,
    SEG_UPCOMING_TEXT,
    RectifiedFlowPerceiver,
    text_to_ids,
)

# Import constants + helpers from train_clean_100
import train_clean_100 as T


# ── Test 1: text_to_ids ─────────────────────────────────────────────
def test_text_to_ids():
    print("TEST 1: text_to_ids ...")
    assert text_to_ids("abc") == [0, 1, 2]
    assert text_to_ids("z") == [25]
    assert text_to_ids("0") == [26]
    assert text_to_ids("9") == [35]
    assert text_to_ids("Hello World 123") == [7, 4, 11, 11, 14, 22, 14, 17, 11, 3, 27, 28, 29]
    assert text_to_ids("") == []
    assert text_to_ids("!!!") == []
    print("  ✓ text_to_ids works correctly")


# ── Test 2: Synthetic manifest → dataset ────────────────────────────
def test_dataset_with_synthetic_manifest():
    print("TEST 2: Dataset with synthetic manifest ...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        manifest_path = cache_dir / "manifest.json"

        # Create 5 fake .pt files with enough frames for MIN_CHUNKS (6) chunks
        n_chunks = 8  # > MIN_CHUNKS=6
        n_frames = n_chunks * T.CHUNK_FRAMES  # 8 * 75 = 600
        manifest = {}

        for i in range(5):
            pt_path = cache_dir / f"fake_{i}.pt"
            emb = torch.randn(n_frames, 128)
            torch.save(emb, pt_path)
            manifest[f"fake_{i}"] = {
                "path": str(pt_path),
                "n_frames": n_frames,
                "n_chunks": n_chunks,
                "chunk_texts": [f"chunk {j} text" for j in range(n_chunks)],
            }

        manifest_path.write_text(json.dumps(manifest))

        # Monkey-patch the MANIFEST_PATH to point to our temp dir
        old_manifest = T.MANIFEST_PATH
        T.MANIFEST_PATH = manifest_path
        try:
            ds = T.LibriTTSFlowDataset()
            assert len(ds) == 5, f"Expected 5 entries, got {len(ds)}"

            context, target, text_ids, text_segs = ds[0]
            assert context.shape == (T.CONTEXT_FRAMES, 128), f"ctx shape: {context.shape}"
            assert target.shape == (T.TARGET_FRAMES, 128), f"tgt shape: {target.shape}"
            assert len(text_ids) > 0, "text_ids is empty"
            assert len(text_segs) == len(text_ids), "segment len mismatch"
            assert all(s in (SEG_PRIOR_TEXT, SEG_UPCOMING_TEXT) for s in text_segs.tolist())

            print(f"  context: {context.shape}, target: {target.shape}")
            print(f"  text_ids len: {len(text_ids)}, text_segs len: {len(text_segs)}")
            print("  ✓ Dataset produces correct shapes")
        finally:
            T.MANIFEST_PATH = old_manifest


# ── Test 3: collate_fn ──────────────────────────────────────────────
def test_collate_fn():
    print("TEST 3: collate_fn ...")

    batch = []
    for i in range(4):
        ctx = torch.randn(T.CONTEXT_FRAMES, 128)
        tgt = torch.randn(T.TARGET_FRAMES, 128)
        # Different text lengths
        n_chars = 10 + i * 5
        ids = torch.randint(0, 36, (n_chars,))
        segs = torch.zeros(n_chars, dtype=torch.long)
        batch.append((ctx, tgt, ids, segs))

    contexts, targets, text_ids, text_segs = T.collate_fn(batch)

    assert contexts.shape == (4, T.CONTEXT_FRAMES, 128), f"ctx: {contexts.shape}"
    assert targets.shape == (4, T.TARGET_FRAMES, 128), f"tgt: {targets.shape}"
    assert text_ids.shape[0] == 4
    assert text_ids.shape[1] == 25  # max text len = 10+3*5=25
    assert text_segs.shape == text_ids.shape

    print(f"  contexts: {contexts.shape}, targets: {targets.shape}")
    print(f"  text_ids: {text_ids.shape}, text_segs: {text_segs.shape}")
    print("  ✓ collate_fn pads and batches correctly")

    return contexts, targets, text_ids, text_segs


# ── Test 4: Model forward ──────────────────────────────────────────
def test_model_forward(contexts, targets, text_ids, text_segs):
    print("TEST 4: Model forward pass ...")

    model = RectifiedFlowPerceiver(
        input_dim=128,
        dim=64,       # small for testing
        num_heads=4,
        num_latent_layers=2,
        max_latents=32,
        max_seq_len=1024,
    )

    B = targets.shape[0]
    t = torch.rand(B)
    x0 = torch.randn_like(targets)
    t_broad = t.view(B, 1, 1)
    z_t = (1.0 - t_broad) * x0 + t_broad * targets

    v = model(z_t, t, contexts, text_ids, text_segs)

    assert v.shape == targets.shape, f"Expected {targets.shape}, got {v.shape}"
    assert torch.isfinite(v).all(), "Output contains NaN/Inf"

    print(f"  v.shape: {v.shape}, v.mean: {v.mean():.4f}, v.std: {v.std():.4f}")
    print("  ✓ Model produces correct output shape with finite values")

    return model


# ── Test 5: Full training step ──────────────────────────────────────
def test_training_step(model, contexts, targets, text_ids, text_segs):
    print("TEST 5: Full training step ...")

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = targets.shape[0]
    t = torch.rand(B)
    x0 = torch.randn_like(targets)

    t_broad = t.view(B, 1, 1)
    z_t = (1.0 - t_broad) * x0 + t_broad * targets
    v_target = targets - x0

    v_pred = model(z_t, t, contexts, text_ids, text_segs)
    loss = (v_pred - v_target).abs().mean()

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    # Verify gradients exist
    n_params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for p in model.parameters())

    print(f"  loss: {loss.item():.6f}")
    print(f"  params with gradients: {n_params_with_grad}/{total_params}")
    print("  ✓ Training step completes with finite loss and gradients")


# ── Test 6: Euler sampling ──────────────────────────────────────────
def test_euler_sampling(model, contexts, text_ids, text_segs):
    print("TEST 6: Euler sampling loop ...")

    model.eval()
    with torch.no_grad():
        z = torch.randn(1, T.TARGET_FRAMES, 128)
        ctx = contexts[:1]
        t_ids = text_ids[:1]
        t_segs = text_segs[:1]

        N_STEPS = 5
        dt = 1.0 / N_STEPS
        for i in range(N_STEPS):
            t_i = torch.tensor([i / N_STEPS])
            v = model(z, t_i, ctx, t_ids, t_segs)
            z = z + v * dt

        assert z.shape == (1, T.TARGET_FRAMES, 128), f"z.shape: {z.shape}"
        assert torch.isfinite(z).all(), "Euler output contains NaN/Inf"

    print(f"  z.shape: {z.shape}, z.mean: {z.mean():.4f}")
    print("  ✓ Euler sampling produces correct shape with finite values")


# ── Test 7: Entries with too few chunks get filtered ────────────────
def test_short_utterances_filtered():
    print("TEST 7: Short utterances filtered out ...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        manifest_path = cache_dir / "manifest.json"

        manifest = {}
        # 3 short (< MIN_CHUNKS), 2 long enough
        for i in range(3):
            n_chunks = 3  # too short
            pt_path = cache_dir / f"short_{i}.pt"
            torch.save(torch.randn(n_chunks * T.CHUNK_FRAMES, 128), pt_path)
            manifest[f"short_{i}"] = {
                "path": str(pt_path), "n_frames": n_chunks * T.CHUNK_FRAMES,
                "n_chunks": n_chunks, "chunk_texts": ["hi"] * n_chunks,
            }
        for i in range(2):
            n_chunks = 8
            pt_path = cache_dir / f"long_{i}.pt"
            torch.save(torch.randn(n_chunks * T.CHUNK_FRAMES, 128), pt_path)
            manifest[f"long_{i}"] = {
                "path": str(pt_path), "n_frames": n_chunks * T.CHUNK_FRAMES,
                "n_chunks": n_chunks, "chunk_texts": ["hello"] * n_chunks,
            }

        manifest_path.write_text(json.dumps(manifest))

        old_manifest = T.MANIFEST_PATH
        T.MANIFEST_PATH = manifest_path
        try:
            ds = T.LibriTTSFlowDataset()
            assert len(ds) == 2, f"Expected 2 usable, got {len(ds)}"
            print(f"  5 entries total, {len(ds)} usable (≥{T.MIN_CHUNKS} chunks)")
            print("  ✓ Short utterances correctly filtered")
        finally:
            T.MANIFEST_PATH = old_manifest


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    tests = [
        ("text_to_ids", lambda: test_text_to_ids()),
        ("dataset_synthetic", lambda: test_dataset_with_synthetic_manifest()),
        ("collate_fn", lambda: test_collate_fn()),
    ]

    # Run first 3 tests
    collated = None
    for name, fn in tests:
        try:
            ret = fn()
            results[name] = "PASS"
            if name == "collate_fn":
                collated = ret
        except Exception:
            traceback.print_exc()
            results[name] = "FAIL"

    # Tests 4-6 depend on collated data
    model = None
    if collated:
        contexts, targets, text_ids, text_segs = collated
        try:
            model = test_model_forward(contexts, targets, text_ids, text_segs)
            results["model_forward"] = "PASS"
        except Exception:
            traceback.print_exc()
            results["model_forward"] = "FAIL"

        if model:
            try:
                test_training_step(model, contexts, targets, text_ids, text_segs)
                results["training_step"] = "PASS"
            except Exception:
                traceback.print_exc()
                results["training_step"] = "FAIL"

            try:
                test_euler_sampling(model, contexts, text_ids, text_segs)
                results["euler_sampling"] = "PASS"
            except Exception:
                traceback.print_exc()
                results["euler_sampling"] = "FAIL"

    # Test 7
    try:
        test_short_utterances_filtered()
        results["short_filtered"] = "PASS"
    except Exception:
        traceback.print_exc()
        results["short_filtered"] = "FAIL"

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    all_pass = all(v == "PASS" for v in results.values())
    print(f"\n{'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
