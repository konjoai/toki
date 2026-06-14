"""Safety-subspace LoRA utilities (Sprint 17 — v1.7.0).

Three complementary approaches from the 2025-2026 safety-preserving LoRA
literature, all validated on 1B-3B model targets (toki's range):

  SaLoRA  (arXiv 2501.01765) — training-time: freeze a pre-computed safety delta
            before task fine-tuning begins so alignment features are not overwritten.
  SPLoRA  (arXiv 2506.18931, TACL) — post-hoc: E-DIEM audit of weight-update
            shifts after fine-tuning completes; flags layers that eroded safety.
  Rank-1  (arXiv 2507.17075) — rank-1 LoRA on middle up_proj layers only;
            zero reasoning tax; minimal intervention for reasoning-capable models.

All torch / peft operations are deferred behind try-import guards.
Functions raise ImportError("requires toki[hf]: pip install toki[hf]") cleanly
when optional deps are absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SafetyLoRAConfig:
    """Safety-preserving extension fields for LoRAConfig.

    Attributes
    ----------
    safety_lora_rank:
        Rank for the frozen safety adapter (0 = disabled). Rank-1 on middle
        up_proj layers is sufficient with zero reasoning tax (arXiv 2507.17075).
    safety_subspace_path:
        Path to a pre-computed safety delta checkpoint (.pt file). When set,
        the delta is applied and those parameters are frozen before task
        fine-tuning begins (SaLoRA approach, arXiv 2501.01765).
    enable_splora_audit:
        When True, run E-DIEM safety-subspace audit after training completes
        (SPLoRA post-hoc check, arXiv 2506.18931).
    splora_threshold:
        E-DIEM normalised Frobenius distance above which a layer is flagged.
        Default 0.15 follows the SPLoRA paper's reported threshold.
    """

    safety_lora_rank: int = 0
    safety_subspace_path: Optional[str] = None
    enable_splora_audit: bool = False
    splora_threshold: float = 0.15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SploraAuditResult:
    """Result of an E-DIEM safety-subspace audit (SPLoRA, arXiv 2506.18931).

    Attributes
    ----------
    flagged_layers:
        Parameter names whose weight-update E-DIEM distance exceeded threshold.
    max_ediem:
        Highest E-DIEM distance observed across all checked layers.
    passed:
        True when no layers were flagged (all updates within safety subspace).
    threshold:
        The E-DIEM threshold used for this audit.
    """

    flagged_layers: List[str]
    max_ediem: float
    passed: bool
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged_layers": list(self.flagged_layers),
            "max_ediem": self.max_ediem,
            "passed": self.passed,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# LoRATrainResult
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainResult:
    """Return value of LoRAFinetuner.train().

    Attributes
    ----------
    training_loss:
        Final training loss from the HF Trainer.
    num_steps:
        Total training steps completed.
    splora_audit:
        E-DIEM safety-subspace audit result, or None when
        enable_splora_audit=False (the default).
    """

    training_loss: float
    num_steps: int
    splora_audit: Optional[SploraAuditResult] = None


# ---------------------------------------------------------------------------
# SaLoRA — load and freeze safety subspace
# ---------------------------------------------------------------------------


def load_safety_subspace(path: str) -> Dict[str, Any]:
    """Load a safety delta checkpoint from disk.

    Raises ``ImportError`` when torch is absent.
    Raises ``FileNotFoundError`` when the checkpoint file does not exist.

    Parameters
    ----------
    path:
        Filesystem path to a PyTorch checkpoint (.pt / .bin) produced by
        saving a safety adapter's state dict.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "load_safety_subspace requires toki[hf]: pip install toki[hf]"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Safety subspace checkpoint not found: {path}")

    state: Dict[str, Any] = torch.load(str(p), map_location="cpu", weights_only=True)
    logger.debug("load_safety_subspace: loaded %d tensors from %s", len(state), path)
    return state


def freeze_safety_adapter(
    model: Any,
    safety_delta: Optional[Dict[str, Any]],
) -> None:
    """Freeze safety-alignment parameters in the model.

    When *safety_delta* is ``None`` this is a complete no-op — no model
    modification occurs and no imports are attempted.

    Otherwise, for each parameter name in *safety_delta* that matches a
    model parameter, the delta tensor is added to the parameter data in-place
    and that parameter is marked ``requires_grad=False``. This prevents
    task fine-tuning from overwriting the safety-alignment features
    (SaLoRA, arXiv 2501.01765).

    Parameters
    ----------
    model:
        Any ``nn.Module`` (typically a peft.PeftModel from prepare_model()).
    safety_delta:
        Parameter state dict from :func:`load_safety_subspace`. Keys are
        parameter names; values are tensors.
    """
    if safety_delta is None:
        return

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "freeze_safety_adapter requires toki[hf]: pip install toki[hf]"
        ) from exc

    matched = 0
    for name, param in model.named_parameters():
        key = name.replace("base_model.model.", "")
        if key not in safety_delta:
            continue
        delta = safety_delta[key].to(param.device)
        if delta.shape != param.shape:
            logger.warning(
                "freeze_safety_adapter: shape mismatch for %r (%s vs %s), skipping",
                name,
                tuple(param.shape),
                tuple(delta.shape),
            )
            continue
        with torch.no_grad():
            param.data += delta
        param.requires_grad_(False)
        matched += 1

    logger.debug(
        "freeze_safety_adapter: froze %d / %d safety-delta tensors",
        matched,
        len(safety_delta),
    )


# ---------------------------------------------------------------------------
# SPLoRA — E-DIEM post-hoc audit
# ---------------------------------------------------------------------------


def _ediem(base_tensor: Any, fine_tuned_tensor: Any) -> float:
    """Compute simplified E-DIEM distance between two tensors.

    E-DIEM (Empirical Dimension-Insensitive Evidence Metric) from SPLoRA
    (arXiv 2506.18931) measures how much a fine-tuned weight update shifts
    the representation away from the safety subspace.

    We use the normalised Frobenius norm of the weight delta as a practical
    approximation when a pre-computed safety-subspace projection is unavailable.
    Returns 0.0 when torch is absent (safe no-op path).
    """
    try:
        import torch as _torch  # noqa: F401
    except ImportError:
        return 0.0

    delta = fine_tuned_tensor.float() - base_tensor.float()
    base_norm = base_tensor.float().norm(p="fro").item()
    if base_norm < 1e-8:
        return 0.0
    return delta.norm(p="fro").item() / base_norm


def splora_audit(
    model: Any,
    base_state: Dict[str, Any],
    *,
    threshold: float = 0.15,
) -> SploraAuditResult:
    """Run an E-DIEM safety-subspace audit of a fine-tuned model.

    Compares each parameter of the fine-tuned model against the pre-training
    base state. Flags layers where the normalised Frobenius distance exceeds
    *threshold*.

    Logs a WARNING for every flagged layer so the CI output is actionable.
    Logs a WARNING when no matching layers are found (misconfigured base_state).

    Parameters
    ----------
    model:
        Fine-tuned model (peft.PeftModel or any nn.Module).
    base_state:
        Pre-training parameter state dict; keys = parameter names,
        values = tensors captured before trainer.train().
    threshold:
        E-DIEM distance above which a layer is flagged.
    """
    try:
        import torch as _torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "splora_audit requires toki[hf]: pip install toki[hf]"
        ) from exc

    flagged: List[str] = []
    max_dist = 0.0
    checked = 0

    current: Dict[str, Any] = {k: v for k, v in model.named_parameters()}

    for name, base_tensor in base_state.items():
        if name not in current:
            continue
        dist = _ediem(base_tensor, current[name].data)
        if dist > max_dist:
            max_dist = dist
        if dist > threshold:
            flagged.append(name)
            logger.warning(
                "SPLoRA audit: %r E-DIEM=%.4f exceeds threshold=%.4f",
                name,
                dist,
                threshold,
            )
        checked += 1

    if checked == 0:
        logger.warning(
            "SPLoRA audit: no matching layers found between model and base_state "
            "— verify that base_state was captured from the same model"
        )

    return SploraAuditResult(
        flagged_layers=flagged,
        max_ediem=max_dist,
        passed=len(flagged) == 0,
        threshold=threshold,
    )
