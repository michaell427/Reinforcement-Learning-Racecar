"""Phase 1 validation: run random agent and verify env works."""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np

from src.env import RacecarEnv
from src.env.track import make_oval_track


def main():
    print("Phase 1 Validation: Random Agent")
    print("=" * 50)
    track = make_oval_track(length=25, width=5)
    env = RacecarEnv(track=track, max_episode_steps=500, render_mode=None)
    obs, info = env.reset(seed=42)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs shape: {obs.shape}, obs: {obs}")
    total_reward = 0.0
    steps = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            print(f"Episode ended: steps={steps}, reward={total_reward:.1f}, lap_count={info.get('lap_count', 0)}")
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
    env.close()
    print("OK: Random agent runs without crashing.")


def test_render():
    """Quick visual check (optional)."""
    track = make_oval_track(length=25, width=5)
    env = RacecarEnv(track=track, max_episode_steps=100, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(20):
        action = np.array([0.0, 0.5], dtype=np.float32)  # straight, half throttle
        env.step(action)
    frame = env.render()
    print(f"Render frame shape: {frame.shape}")
    env.close()
    return frame


if __name__ == "__main__":
    main()
    print("\nTesting render...")
    frame = test_render()
    print("Validation complete.")
