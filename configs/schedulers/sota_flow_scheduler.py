import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler


class SDXLAnime_UltimateDPM3MFlow(DPMSolverMultistepScheduler):
    """
    diffusers==0.35.2 compatible.
    DPM-Solver++ 3M core, Karras sigmas, trailing spacing.
    Safely warps the *index progression* of sigmas in the middle band and
    then recomputes timesteps from the new sigmas (so UNet conditioning matches).
    """

    def __init__(self, *args, gamma: float = 1.00, sde_strength: float = 0, **kwargs):
        self.gamma = float(gamma)
        self.sde_strength = float(sde_strength)

        # Force the intended DPMSolver++ 3M setup
        kwargs["algorithm_type"] = "dpmsolver++"
        kwargs["solver_order"] = 2
        kwargs["lower_order_final"] = True
        kwargs["use_karras_sigmas"] = True
        kwargs["final_sigmas_type"] = "sigma_min"
        kwargs["timestep_spacing"] = "trailing"

        super().__init__(*args, **kwargs)

    def _enforce_strictly_nonincreasing(self, sig: np.ndarray) -> np.ndarray:
        """Make sig strictly non-increasing (except it may end at 0)."""
        sig = sig.astype(np.float32, copy=False)

        # Ensure penultimate >= last (important when last is sigma_min and we scaled others)
        if sig.shape[0] >= 2 and sig[-2] < sig[-1]:
            sig[-2] = sig[-1]

        for i in range(1, sig.shape[0]):
            if sig[i] > sig[i - 1]:
                # push down to the next representable float toward 0 (keeps direction correct)
                sig[i] = np.nextafter(sig[i - 1], np.float32(0.0)).astype(np.float32)
            elif sig[i] == sig[i - 1] and sig[i] != 0.0:
                # avoid exact duplicates which can stall a step (h ~ 0)
                sig[i] = np.nextafter(sig[i], np.float32(0.0)).astype(np.float32)

        return sig

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device=None,
        mu=None,
        timesteps=None,
    ):
        # Build the base schedule first (includes sigma_last append + base timesteps)
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
            timesteps=timesteps,
        )

        # If user provided custom timesteps, don't touch anything.
        if timesteps is not None:
            return

        sig = self.sigmas.detach().cpu().numpy().astype(np.float32)
        if sig.ndim != 1:
            sig = sig.reshape(-1)

        # sig length is num_inference_steps + 1, last entry is sigma_last
        n_infer = int(sig.shape[0] - 1)
        if n_infer < 2:
            return

        # Middle 60% (exclude the final appended sigma)
        mid_start = int(n_infer * 0.20)
        mid_end = int(n_infer * 0.80)
        mid_start = max(0, min(mid_start, n_infer - 1))
        mid_end = max(mid_start + 2, min(mid_end, n_infer))  # at least 2 points

        # 1) Gentle "anime gamma" by warping *index* (not sigma values)
        if abs(self.gamma - 1.0) > 1e-6 and (mid_end - mid_start) >= 2:
            seg = sig[mid_start:mid_end].copy()
            t = np.linspace(0.0, 1.0, seg.size, dtype=np.float32)
            tw = (t ** np.float32(self.gamma)).astype(np.float32)
            # seg is monotone decreasing; sampling it with increasing tw preserves monotonicity
            sig[mid_start:mid_end] = np.interp(tw, t, seg).astype(np.float32)

        # 2) Very light global sigma scaling (keep final sigma exactly unchanged)
        sigma_last = sig[-1]
        if self.sde_strength != 0.0:
            scale = np.float32(1.0 - self.sde_strength)
            sig[:-1] = (sig[:-1] * scale).astype(np.float32)
            sig[-1] = sigma_last

        # 3) Hard safety: preserve correct direction for DPM updates
        sig = self._enforce_strictly_nonincreasing(sig)

        # 4) Recompute timesteps from sigmas so UNet conditioning matches the new schedule
        #    (same primitives as diffusers==0.35.2 DPMSolverMultistepScheduler.set_timesteps)
        train_sigmas = (
            (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        log_sigmas = np.log(train_sigmas)

        # Map each inference sigma (excluding sigma_last) to a timestep
        ts = []
        for s in sig[:-1]:
            t = self._sigma_to_t(np.array([s], dtype=np.float32), log_sigmas).item()
            ts.append(t)
        timesteps_new = np.array(ts, dtype=np.float32)

        # Match diffusers rounding behavior for non-cosine schedules
        if self.config.beta_schedule != "squaredcos_cap_v2":
            timesteps_new = np.rint(timesteps_new)

        timesteps_new = timesteps_new.astype(np.int64)

        # Commit
        self.sigmas = torch.from_numpy(sig).to("cpu")  # keep on CPU like upstream
        self.timesteps = torch.from_numpy(timesteps_new).to(
            device=device, dtype=torch.int64
        )
        self.num_inference_steps = len(timesteps_new)

        # Reset solver state (same as upstream set_timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
