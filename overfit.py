"""
Rectified-flow overfitter: memorise a single 10 s clip in the
continuous EnCodec latent space.

Usage:
    uv run python overfit.py
"""

import torch
from codec import Codec
from perceiver import RectifiedFlowPerceiver

AUDIO_PATH = "rice_speech_10s.wav"
STEPS = 5000
LR = 1e-3
PRINT_EVERY = 100

# ------------------------------------------------------------------
# 1. Encode the target clip to continuous embeddings
# ------------------------------------------------------------------
print("Loading codec & encoding audio …")
codec = Codec()
x1 = codec.encode_continuous(AUDIO_PATH)  # [1, 128, 750]
x1 = x1.permute(0, 2, 1)                  # [1, 750, 128]  (B, seq, dim)
print(f"Target shape: {x1.shape}")

# ------------------------------------------------------------------
# 2. Build model
# ------------------------------------------------------------------
model = RectifiedFlowPerceiver(
    input_dim=128,
    dim=256,
    num_heads=8,
    num_latent_layers=6,
    max_latents=128,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimiser = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------
# 3. Training loop — rectified flow on a single sample
# ------------------------------------------------------------------
print(f"Overfitting for {STEPS} steps …")
for step in range(1, STEPS + 1):
    # Random time t ∈ [0, 1]
    t = torch.rand(1)  # (B=1,)

    # Random noise x_0
    x0 = torch.randn_like(x1)

    # Mix: z_t = (1 - t) * x_0 + t * x_1
    t_broad = t.view(1, 1, 1)  # broadcast over (B, seq, dim)
    z_t = (1.0 - t_broad) * x0 + t_broad * x1

    # Target velocity: v = x_1 - x_0
    v_target = x1 - x0

    # Predict
    v_pred = model(z_t, t)

    # L1 loss
    loss = (v_pred - v_target).abs().mean()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if step % PRINT_EVERY == 0 or step == 1:
        print(f"  step {step:5d}/{STEPS}  loss = {loss.item():.6f}")

# ------------------------------------------------------------------
# 4. Sample: Euler integration from noise → data
# ------------------------------------------------------------------
print("\nSampling with 100-step Euler …")
N_EULER = 100
dt = 1.0 / N_EULER

model.eval()
with torch.no_grad():
    z = torch.randn_like(x1)  # start from pure noise
    for i in range(N_EULER):
        t_i = torch.tensor([i / N_EULER])
        v = model(z, t_i)
        z = z + v * dt

# z is now our predicted x_1 → back to [1, 128, 750]
sample = z.permute(0, 2, 1)  # [1, 128, 750]

print("Decoding sample to audio …")
codec.decode_continuous(sample, "overfit_sample.mp3")
print("Saved → overfit_sample.mp3")
