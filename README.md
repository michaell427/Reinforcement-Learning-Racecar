# Reinforcement Learning Racecar

A 2D racing simulation with PPO training (Gymnasium, Stable-Baselines3).

## Setup

```bash
pip install -r requirements.txt
```

## Phase 1 Validation

Run a random agent to verify the environment works:

```bash
python scripts/validate_env.py
```

## Project Structure

- `src/env/` — Gymnasium environment, physics, track, rewards
- `scripts/` — Training and evaluation scripts
- `ROADMAP.md` — Full implementation plan
