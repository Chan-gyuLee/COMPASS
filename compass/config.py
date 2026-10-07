"""Explicit, serializable settings for the public reference implementation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SteeringConfig:
    anchor_time: float = 0.5
    anchor_steps: int = 25
    sampling_steps: int = 50
    optimization_steps: int = 100
    learning_rate: float = 0.015
    weight_ca: float = 1.0
    weight_cb: float = 2.0
    weight_orientation: float = 1.0
    weight_regularization: float = 1.0
    delta_clamp: float = 0.5
    gradient_clip: float = 1.0
    orientation_mode: str = "relative"

    def __post_init__(self):
        if not math.isfinite(self.anchor_time) or not 0.001 < self.anchor_time < 0.99:
            raise ValueError("anchor_time must be between 0.001 and 0.99")
        for name in ("anchor_steps", "sampling_steps", "optimization_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < (
                1 if name == "optimization_steps" else 2
            ):
                raise ValueError(
                    f"{name} must be an integer >= {1 if name == 'optimization_steps' else 2}"
                )
        for name in ("learning_rate", "delta_clamp", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_ca",
            "weight_cb",
            "weight_orientation",
            "weight_regularization",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.weight_ca + self.weight_cb + self.weight_orientation == 0:
            raise ValueError("At least one geometric loss weight must be positive")
        if self.orientation_mode not in {"relative", "legacy_absolute"}:
            raise ValueError("orientation_mode must be relative or legacy_absolute")
