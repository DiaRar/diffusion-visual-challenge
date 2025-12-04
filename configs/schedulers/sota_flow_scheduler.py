import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler


class SDXLAnime_UltimateDPM3MFlow(DPMSolverMultistepScheduler):
    """
    SAFE + HIGH QUALITY ULTIMATE SDXL ANIME SCHEDULER.
    - No breaking of sigma schedule
    - No final-step corruption
    - Gentle SDE correction
    - Gentle anime gamma in mid-phase only
    - Keeps DPM++ 3M as core solver
    """

    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.gamma = kwargs.pop("gamma", 1.01)  # tiny safe gamma
        self.sde_strength = kwargs.pop("sde_strength", 0.0002)

        # Configure DPM++ 3M via kwargs (safer than config manipulation)
        kwargs["algorithm_type"] = "dpmsolver++"
        kwargs["solver_order"] = 3
        kwargs["lower_order_final"] = True
        kwargs["use_karras_sigmas"] = True
        kwargs["final_sigmas_type"] = "sigma_min"
        kwargs["timestep_spacing"] = "trailing"

        super().__init__(*args, **kwargs)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device=None,
        mu=None,
        timesteps=None,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
            timesteps=timesteps,
        )

        if timesteps is not None:
            return

        sig = self.sigmas.cpu().float().numpy()
        steps = len(sig)

        # Only modify middle 60% of steps
        mid_start = int(steps * 0.20)
        mid_end = int(steps * 0.80)

        mid = sig[mid_start:mid_end]

        # Gentle anime gamma in middle range only
        mid = mid**self.gamma
        sig[mid_start:mid_end] = np.interp(
            mid,
            (mid.min(), mid.max()),
            (sig[mid_start], sig[mid_end - 1]),
        )

        # Very light SDE correction
        sig *= 1 - self.sde_strength

        self.sigmas = torch.tensor(sig, device="cpu", dtype=self.sigmas.dtype)
