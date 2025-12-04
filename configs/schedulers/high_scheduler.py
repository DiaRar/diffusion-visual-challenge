# diffusers==0.35.2
from __future__ import annotations

from typing import Optional, Union

import torch
from diffusers import DPMSolverMultistepScheduler


class SDXLAnime_BestHQScheduler(DPMSolverMultistepScheduler):
    """
    Best-quality SDXL-base scheduler (still images), diffusers==0.35.2.

    Philosophy:
      - NO custom sigma warping (that's what was scuffing quality)
      - Use stable, high-quality DPM++ 2M configuration
      - SDXL-specific stabilization knobs available (lu_lambdas / karras)
      - Future-proof hooks for video experiments without touching solver math
    """

    def __init__(
        self,
        *args,
        # Recommended for SDXL stability/quality when using DPM++ (esp. <50 steps),
        # also fine at 60-100 steps as a strong default.
        use_lu_lambdas: bool = True,
        # Optional alternative; keep False by default if you want the most "vanilla SDXL" behavior.
        use_karras_sigmas: bool = False,
        # For images, ODE DPM++ is typically the crispest.
        algorithm_type: str = "dpmsolver++",
        **kwargs,
    ):
        # Remove diffusers private config keys if passed via from_config
        kwargs.pop("_class_name", None)
        kwargs.pop("_diffusers_version", None)

        kwargs.setdefault(
            "algorithm_type", algorithm_type
        )  # "dpmsolver++" or "sde-dpmsolver++"
        kwargs.setdefault("solver_order", 2)  # best for CFG / guided sampling
        kwargs.setdefault("solver_type", "midpoint")  # stable + sharp
        kwargs.setdefault("lower_order_final", True)
        kwargs.setdefault("euler_at_final", False)
        kwargs.setdefault(
            "final_sigmas_type", "zero"
        )  # fully denoise (don’t stop early)
        kwargs.setdefault("final_sigmas_type", "zero")        # fully denoise (don’t stop early)
        kwargs.setdefault("timestep_spacing", "linspace")

        # SDXL stabilization schedule choice (only one should be True)
        kwargs.setdefault("use_lu_lambdas", bool(use_lu_lambdas))
        kwargs.setdefault("use_karras_sigmas", bool(use_karras_sigmas))

        super().__init__(*args, **kwargs)

    # ---- future-proof hooks (no-op now) ----
    # Later for video: you can override these to inject temporal conditioning,
    # per-step CFG schedules, cached features, etc., without modifying solver updates.
    def hook_before_step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        return model_output

    def hook_after_step(
        self,
        prev_sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        return prev_sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        model_output = self.hook_before_step(model_output, timestep, sample)
        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        )
        if return_dict:
            out.prev_sample = self.hook_after_step(out.prev_sample, timestep, sample)
            return out
        prev_sample = out[0]
        prev_sample = self.hook_after_step(prev_sample, timestep, sample)
        return (prev_sample,)
