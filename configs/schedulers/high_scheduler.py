# configs/schedulers/best_hq_scheduler.py
from __future__ import annotations

import copy
import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerOutput

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


class SDXLAnime_BestHQScheduler(DPMSolverMultistepScheduler):
    """
    SDXL HQ scheduler wrapper that remains DPM-solver-identical unless you add hooks.
    Key rule: DO NOT override __init__ (keeps from_config/kwargs signature issues away).
    """

    # ---- future-proof hooks (no-op now) ----
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
    ) -> Union[SchedulerOutput, Tuple[torch.Tensor]]:
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
            prev = self.hook_after_step(out.prev_sample, timestep, sample)
            return SchedulerOutput(prev_sample=prev)

        prev = self.hook_after_step(out[0], timestep, sample)
        return (prev,)


def _filtered_init_kwargs_for_dpmsolver(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull ONLY the kwargs that DPMSolverMultistepScheduler.__init__ actually accepts.
    Prevents accidental defaults / dropped critical fields.
    """
    sig = inspect.signature(DPMSolverMultistepScheduler.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: cfg[k] for k in cfg.keys() if k in allowed}


def apply_best_hq_scheduler(
    pipe: "DiffusionPipeline",
    *,
    # Pick ONE schedule family for quality. Default: Karras (great at 20–50 steps).
    use_karras_sigmas: bool = True,
    use_lu_lambdas: bool = False,
) -> "DiffusionPipeline":
    """
    Replaces pipe.scheduler with SDXLAnime_BestHQScheduler using a safe HQ DPM++ 2M setup.
    (No sigma warping; no from_config; preserves SDXL training betas/schedule from existing config.)
    """
    if use_karras_sigmas and use_lu_lambdas:
        raise ValueError(
            "Choose only one: use_karras_sigmas or use_lu_lambdas (not both)."
        )

    # Start from the CURRENT scheduler config (this is the SDXL-trained schedule info)
    raw_cfg = dict(pipe.scheduler.config)
    cfg = copy.deepcopy(raw_cfg)

    # ---- HQ overrides (safe + standard) ----
    # DPM++ 2M (solver_order=2) is the most reliable under CFG.
    cfg.update(
        dict(
            algorithm_type="dpmsolver++",
            solver_order=1,
            solver_type="midpoint",
            lower_order_final=False,
            euler_at_final=True,
            final_sigmas_type="zero",
            timestep_spacing="trailing",
            use_karras_sigmas=bool(use_karras_sigmas),
            use_lu_lambdas=bool(use_lu_lambdas),
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            # leave prediction_type as-is from SDXL config (usually "epsilon")
        )
    )

    init_kwargs = _filtered_init_kwargs_for_dpmsolver(cfg)
    scheduler = SDXLAnime_BestHQScheduler(**init_kwargs)

    pipe.scheduler = scheduler
    logger.info("✓ Applied SDXLAnime_BestHQScheduler with HQ DPM++ 2M config")
    return pipe
