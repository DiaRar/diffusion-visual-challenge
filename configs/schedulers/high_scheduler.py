import numpy as np
from diffusers import DPMSolverMultistepScheduler


class HighQualityDPMScheduler(DPMSolverMultistepScheduler):
    """
    High-Quality DPM++ 2M scheduler optimized for SDXL Anime.

    Provides:
    - Better high-frequency detail reproduction
    - Stable color gradients (no chroma blowout)
    - Cleaner final step (no mush / over-denoise)
    - Safe timestep spacing for SDXL latents
    - Gentle anime micro-contrast boost

    Best for:
    - General anime generation (20-30 steps)
    - Balanced quality/speed
    - Stable, reproducible results
    """

    def __init__(self, *args, **kwargs):
        # ================================================================
        # 1. Solver Settings (DPM++ 2M - balanced quality/speed)
        # ================================================================
        kwargs["algorithm_type"] = "dpmsolver++"
        kwargs["solver_order"] = 2  # DPM++ 2M
        kwargs["lower_order_final"] = True  # Stable final transition

        # ================================================================
        # 2. Sigma Schedule (Karras noise schedule)
        # ================================================================
        kwargs["use_karras_sigmas"] = True  # Industry-standard Karras sigmas

        # ================================================================
        # 3. Timestep Spacing
        # ================================================================
        # Trailing = exponential spacing, preserves early detail better
        kwargs["timestep_spacing"] = "trailing"

        # ================================================================
        # 4. Final Sigma Handling
        # ================================================================
        kwargs["final_sigmas_type"] = "sigma_min"  # No color burning
        kwargs["thresholding"] = False  # Avoid SDXL artifacting

        # ================================================================
        # 5. Anime Detail Boost (Optional)
        # ================================================================
        # Slight gamma curve for micro-contrast enhancement
        anime_gamma = kwargs.pop("anime_gamma", 1.02)
        self.anime_gamma = anime_gamma

        super().__init__(*args, **kwargs)

    # --------------------------------------------------------------------
    # Apply small anime gamma to the sigma curve (helps fine detail)
    # --------------------------------------------------------------------
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        super().set_timesteps(num_inference_steps, device=device, **kwargs)

        # Apply detail-preserving gamma, only if num steps >= 12
        if num_inference_steps >= 12:
            sigmas = self.sigmas.cpu().float().numpy()
            sigmas = sigmas**self.anime_gamma

            # Re-normalize to the same range
            sigmas = np.interp(
                sigmas,
                (sigmas.min(), sigmas.max()),
                (self.sigmas.min().cpu(), self.sigmas.max().cpu()),
            )

            self.sigmas = type(self.sigmas)(sigmas).to(device)
