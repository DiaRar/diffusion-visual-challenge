# SDXL Temporal Adapter Video Pipeline

This document describes how to extend **SDXL Base** with a custom **temporal adapter** (no AnimateDiff), prepare a training dataset, train the adapter on an **A100 20GB**, and run inference with your custom **SDXLAnime_BestHQScheduler** (DPM++ 2M Karras compatible).

---

# 1. Scheduler + Temporal Adapter Integration

Your existing `SDXLAnime_BestHQScheduler` already provides:

* Full **DPM++ 2M Karras math** compatibility.
* Clean `hook_before_step` / `hook_after_step` points.

The scheduler does **not** need modifications for temporal adapters.
The UNet will now produce temporally-aware noise predictions.

## Optional: correlated noise smoothing hook

```python
def hook_before_step(self, model_output, timestep, sample):
    if sample.ndim == 5:  # [B, F, C, H, W]
        alpha = 0.2
        sample = sample.clone()
        sample[:, 1:] = alpha * sample[:, :-1] + (1 - alpha) * sample[:, 1:]
    return model_output
```

---

# 2. Modify SDXL UNet: Add Temporal Adapter

## Design goals

* Keep **full SDXL weight compatibility**.
* Add lightweight, zero-init **TemporalSelfAttention**.
* Insert it **between self-attn and cross-attn** inside `BasicTransformerBlock`.

## TemporalSelfAttention module

```python
class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, max_frames=24):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(max_frames, dim) * 0.02)
        self.gate = nn.Parameter(torch.zeros(1))  # zero-init

    def forward(self, x, num_frames: int):
        Bf, S, C = x.shape
        F = num_frames
        B = Bf // F

        x = x.view(B, F, S, C)
        x_t = x.permute(0, 2, 1, 3).reshape(B*S, F, C)

        qkv = self.to_qkv(x_t).chunk(3, dim=-1)
        q, k, v = [t.view(B*S, F, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        pe = self.pos_emb[:F].unsqueeze(0).unsqueeze(0)
        q = q + pe
        k = k + pe

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B*S, F, C)
        out = self.proj(attn).view(B, S, F, C).permute(0, 2, 1, 3).reshape(B*F, S, C)

        return self.gate * out
```

## Temporal-enhanced Transformer block

```python
class TemporalBasicTransformerBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = kwargs.get("dim", self.norm1.normalized_shape[0])
        self.temporal_attn = TemporalSelfAttention(dim, num_heads=self.attn1.num_heads)

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        num_frames = kwargs.get("num_frames", 1)

        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.temporal_attn(hidden_states, num_frames)
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states
```

## Replace SDXL blocks with temporal ones

```python
for name, module in unet.named_modules():
    if isinstance(module, BasicTransformerBlock):
        replace_with_temporal_block(module)
```

---

# 3. Dataset for Temporal Training

Use **anime video clips (8–24 frames)** with consistent motion and no scene cuts.

### Recommended research-safe datasets

* **Sakuga-42M** (cartoon/animation sequences; CC BY-NC-SA)
* **Anita Dataset** (anime animation sequences; CC BY / CC BY-NC-SA)

### Preprocessing

1. Extract frames:

```bash
ffmpeg -i input.mp4 -vf "fps=12,scale=512:-1:force_original_aspect_ratio=decrease" frames/%06d.png
```

2. Encode frames using SDXL VAE → save as latent tensors.
3. Save clip as `.npz`:

   * `latents [F,4,H/8,W/8]`
   * `text_embeds [77,2048]`

---

# 4. Training on A100 20GB

## What to train

* **Freeze**: SDXL UNet base weights, VAE, text encoders.
* **Train**: only `temporal_attn` parameters.

```python
for n, p in unet.named_parameters():
    p.requires_grad = "temporal_attn" in n
```

## Training config

```
batch_size: 2
num_frames: 8
resolution: 768
grad_accum: 8
fp16: true
lr: 1e-4
optimizer: AdamW(beta1=0.9, beta2=0.999, weight_decay=1e-4)
```

## Training loop (simplified)

```python
for batch in dataloader:
    latents, text_embeds = batch["latents"], batch["text_embeds"]
    noise = torch.randn_like(latents)
    t = torch.randint(0, scheduler.num_train_timesteps, (1,), device=latents.device)

    noisy = scheduler.add_noise(latents, noise, t)
    pred = unet(noisy, t, encoder_hidden_states=text_embeds, num_frames=F).sample

    loss = F.mse_loss(pred, noise)
    loss.backward()
```

Train ~100k steps at 768, then ~30k at 1024.

---

# 5. Inference with Your Scheduler

```python
pipe.scheduler = SDXLAnime_BestHQScheduler(**pipe.scheduler.config)
latents = torch.randn((frames, 4, H//8, W//8)).to(device)

for t in pipe.scheduler.timesteps:
    noise_pred = unet(latents, t, encoder_hidden_states=embeds, num_frames=frames).sample
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
```

## Extra quality trick: correlated initial noise

```python
base = torch.randn(1,4,H//8,W//8)
latents = base.repeat(frames,1,1,1)
latents += 0.05 * torch.randn_like(latents)
```

---

# 6. Saving Adapter Weights

```python
state = {k:v for k,v in unet.state_dict().items() if "temporal_attn" in k}
torch.save(state, "temporal_adapter.safetensors")
```

Load later with:

```python
unet.load_state_dict(torch.load("temporal_adapter.safetensors"), strict=False)
```

---

# 7. Pipeline Summary

| Component | Implementation                              |
| --------- | ------------------------------------------- |
| Backbone  | SDXL Base 1.0                               |
| UNet Mods | TemporalSelfAttention (zero-init gated)     |
| Training  | Freeze base, train adapter only             |
| Dataset   | Sakuga-42M, Anita, curated anime clips      |
| Objective | Min-SNR MSE (noise prediction)              |
| Scheduler | SDXLAnime_BestHQScheduler (DPM++ 2M Karras) |
| Inference | Correlated noise + temporal UNet            |
| Output    | High-quality 2–6s anime videos              |

---

If you'd like, I can now generate **drop-in code files** (`temporal_adapter.py`, `training_loop.py`, `dataset.py`, `patch_unet.py`) that match this design exactly.
