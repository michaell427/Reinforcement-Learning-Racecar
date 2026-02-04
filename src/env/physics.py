"""Custom physics: steering, throttle, and kinematic bicycle model."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    """Physics parameters for the racecar."""

    max_speed: float = 15.0
    max_acceleration: float = 8.0
    max_steering_angle: float = np.pi / 6  # 30 degrees
    wheelbase: float = 1.0
    drag_coef: float = 0.1  # velocity-dependent drag
    dt: float = 0.05  # simulation timestep (seconds)


def step_kinematic(
    x: float,
    y: float,
    theta: float,
    speed: float,
    steering: float,
    throttle: float,
    config: PhysicsConfig,
) -> tuple[float, float, float, float]:
    """
    Advance state by one timestep using kinematic bicycle model.

    Args:
        x, y: position
        theta: heading angle (radians)
        speed: current speed (scalar, signed for direction)
        steering: steering input in [-1, 1] (normalized)
        throttle: throttle input in [0, 1]
        config: physics parameters

    Returns:
        (x_new, y_new, theta_new, speed_new)
    """
    dt = config.dt
    # Map throttle to acceleration (0 -> -decel, 1 -> max_accel)
    accel = (throttle * 2 - 1) * config.max_acceleration
    # Map steering to steering angle
    delta = steering * config.max_steering_angle
    # Drag: opposes velocity
    drag = -config.drag_coef * speed * abs(speed)
    speed_new = speed + (accel + drag) * dt
    speed_new = np.clip(speed_new, -config.max_speed * 0.5, config.max_speed)
    # Angular velocity from ackermann/bicycle model: omega = v * tan(delta) / L
    if abs(config.wheelbase) > 1e-10:
        omega = (speed_new * np.tan(delta)) / config.wheelbase
    else:
        omega = 0.0
    theta_new = theta + omega * dt
    # Normalize theta to [-pi, pi]
    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
    # Position update
    x_new = x + speed_new * np.cos(theta_new) * dt
    y_new = y + speed_new * np.sin(theta_new) * dt
    return float(x_new), float(y_new), float(theta_new), float(speed_new)
