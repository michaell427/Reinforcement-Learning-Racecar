"""Reward shaping: lap progress, on-track, speed, completion."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.env.track import Track


@dataclass
class RewardConfig:
    """Reward weights and scaling."""

    progress_weight: float = 1.0
    off_track_penalty: float = -2.0
    on_track_bonus: float = 0.1
    speed_weight: float = 0.05
    lap_completion_bonus: float = 10.0
    normalize_by_dt: bool = False  # scale by timestep; False keeps rewards O(1)


def compute_reward(
    progress_delta: float,
    is_on_track: bool,
    speed: float,
    max_speed: float,
    lap_completed: bool,
    config: RewardConfig,
    dt: float = 0.05,
) -> float:
    """
    Compute step reward from components.

    Args:
        progress_delta: change in lap progress this step (0 to 1)
        is_on_track: whether car is within track boundaries
        speed: current speed
        max_speed: max possible speed (for normalization)
        lap_completed: whether a full lap was completed this step
        config: reward weights
        dt: timestep (for normalization)
    """
    r = 0.0
    # Lap progress (main signal)
    r += config.progress_weight * progress_delta
    # On/off track
    if is_on_track:
        r += config.on_track_bonus
    else:
        r += config.off_track_penalty
    # Speed bonus (encourage moving forward when on track)
    if max_speed > 0 and is_on_track:
        r += config.speed_weight * (speed / max_speed)
    # Sparse lap completion bonus
    if lap_completed:
        r += config.lap_completion_bonus
    if config.normalize_by_dt and dt > 0:
        r /= dt
    return float(r)
