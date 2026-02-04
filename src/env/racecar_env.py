"""Gymnasium racing environment with custom physics and reward shaping."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from src.env.physics import PhysicsConfig, step_kinematic
from src.env.rewards import RewardConfig, compute_reward
from src.env.track import Track, make_oval_track


class RacecarEnv(gym.Env, EzPickle):
    """
    2D racing environment: drive around a track using steering and throttle.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        track: Track | None = None,
        physics_config: PhysicsConfig | None = None,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int = 2000,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        EzPickle.__init__(self, track, physics_config, reward_config, max_episode_steps, render_mode, seed)
        super().__init__()
        self.track = track or make_oval_track()
        self.physics = physics_config or PhysicsConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Action: [steering, throttle], both in [-1, 1] and [0, 1] respectively
        # Normalize to steering in [-1,1] and throttle in [-1,1] for symmetric Box
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: progress, lateral_offset_norm, heading_error_norm, speed_norm, cos(θ), sin(θ)
        self.observation_space = spaces.Box(
            low=np.array([0, -2, -1, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 2, 1, 1, 1, 1], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state: np.ndarray  # (x, y, theta, speed)
        self._progress = 0.0
        self._last_progress = 0.0
        self._lap_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        pos, heading = self.track.sample_start_pose(self._rng)
        speed = 0.0
        self._state = np.array([pos[0], pos[1], heading, speed], dtype=np.float64)
        self._progress = self.track.progress(pos)
        self._last_progress = self._progress
        self._lap_count = 0
        self._step_count = 0
        obs = self._get_obs()
        info = {"progress": self._progress, "lap_count": self._lap_count}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        steering, throttle = float(action[0]), float(np.clip(action[1], 0, 1))
        x, y, theta, speed = self._state
        x_new, y_new, theta_new, speed_new = step_kinematic(
            x, y, theta, speed, steering, throttle, self.physics
        )
        self._state = np.array([x_new, y_new, theta_new, speed_new])
        self._step_count += 1

        pos_new = np.array([x_new, y_new])
        self._last_progress = self._progress
        self._progress = self.track.progress(pos_new)
        progress_delta = self._progress - self._last_progress
        # Handle lap wrap (e.g. 0.95 -> 0.02)
        if progress_delta < -0.5:
            progress_delta += 1.0
        elif progress_delta > 0.5:
            progress_delta -= 1.0
        lap_completed = progress_delta < 0 and self._last_progress > 0.5
        if lap_completed:
            self._lap_count += 1

        is_on_track = self.track.is_on_track(pos_new)
        reward = compute_reward(
            progress_delta=progress_delta,
            is_on_track=is_on_track,
            speed=speed_new,
            max_speed=self.physics.max_speed,
            lap_completed=lap_completed,
            config=self.reward_cfg,
            dt=self.physics.dt,
        )

        terminated = not is_on_track
        truncated = self._step_count >= self.max_episode_steps
        obs = self._get_obs()
        info = {
            "progress": self._progress,
            "lap_count": self._lap_count,
            "is_on_track": is_on_track,
            "lap_completed": lap_completed,
        }
        if self.render_mode == "human" and self._step_count % 2 == 0:
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        x, y, theta, speed = self._state
        pos = np.array([x, y])
        lateral = self.track.lateral_offset(pos)
        track_heading = self.track.heading_at_progress(self._progress)
        heading_error = theta - track_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        half_w = self.track.width / 2
        lateral_norm = np.clip(lateral / (half_w + 1e-6), -2, 2)
        heading_norm = np.clip(heading_error / np.pi, -1, 1)
        speed_norm = np.clip(speed / self.physics.max_speed, 0, 1)
        return np.array(
            [
                self._progress,
                lateral_norm,
                heading_norm,
                speed_norm,
                np.cos(theta),
                np.sin(theta),
            ],
            dtype=np.float32,
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.patches import Circle, Rectangle
        except ImportError:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        waypoints = self.track.waypoints
        half_w = self.track.width / 2
        ax.plot(waypoints[:, 0], waypoints[:, 1], "k-", lw=2, label="Centerline")
        # Inner/outer bounds (approximate with offset)
        n = len(waypoints)
        tangents = np.roll(waypoints, -1, axis=0) - waypoints
        if self.track.loop:
            tangents[-1] = waypoints[0] - waypoints[-1]
        tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        inner = waypoints - half_w * normals
        outer = waypoints + half_w * normals
        ax.plot(inner[:, 0], inner[:, 1], "b--", alpha=0.6)
        ax.plot(outer[:, 0], outer[:, 1], "b--", alpha=0.6)
        x, y, theta, speed = self._state
        circle = Circle((x, y), 0.3, color="red", fill=True, zorder=10)
        ax.add_patch(circle)
        ax.plot([x, x + np.cos(theta)], [y, y + np.sin(theta)], "r-", lw=2)
        ax.set_aspect("equal")
        ax.set_xlim(waypoints[:, 0].min() - 5, waypoints[:, 0].max() + 5)
        ax.set_ylim(waypoints[:, 1].min() - 5, waypoints[:, 1].max() + 5)
        ax.set_title(f"Step {self._step_count} | Progress {self._progress:.2f} | Lap {self._lap_count}")
        plt.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    def close(self) -> None:
        pass
