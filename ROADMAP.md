# Reinforcement Learning Racecar — Implementation Roadmap

A phased plan to build a 2D racing simulation with PPO training, curriculum learning, and evaluation infrastructure — matching the resume description.

---

## Phase 1: Core Environment (Gymnasium API + Physics)

**Goal:** A working Gymnasium environment where an agent can receive observations, take actions, and get rewards.

### 1.1 Project Setup
- [x] Create `requirements.txt`: `gymnasium`, `numpy`, `torch`, `stable-baselines3`, `tensorboard`, `matplotlib` (for visualization)
- [x] Create project structure (see below)
- [ ] Set up basic `pyproject.toml` or `setup.py` if packaging needed

### 1.2 2D Racing Environment (Gymnasium)
- [x] Implement `RacecarEnv(gymnasium.Env)` with `step()`, `reset()`, `render()`
- [x] Define **observation space**: pose (x, y, θ), velocity (vx, vy, ω), and track-relative features (see 1.3)
- [x] Define **action space**: `Box` or `Discrete` for steering + throttle
- [x] Implement **termination/truncation**: off-track, lap completion, max steps

### 1.3 Custom Physics
- [x] **Kinematic model**: simple bicycle/ackermann model for steering
- [x] **Throttle → acceleration** and **steering → angular velocity** mapping
- [x] **Friction/drag** (optional but recommended for realism)
- [x] Track boundaries: line segments or polygon; point-in-polygon / distance checks

### 1.4 Track Representation
- [x] Define track as centerline + width (or inner/outer borders)
- [x] Track-relative features: distance to centerline, angle to track direction, progress along lap (0–1)
- [x] Compute progress from nearest point on centerline or spline

### 1.5 Reward Shaping
- [x] **Lap progress reward**: Δ(progress) per step
- [x] **Staying on track**: penalty for off-track (or reward for on-track)
- [x] **Speed bonus**: optional to encourage faster driving
- [x] **Sparse bonus**: lap completion
- [x] Keep rewards bounded and normalized for stable training

### 1.6 Validation
- [x] Random agent runs without crashing
- [x] Manual debugging: render environment, check obs/rew shapes
- [ ] Unit tests for physics (e.g., straight line, turning radius)

---

## Phase 2: PPO Training Pipeline

**Goal:** Train a PPO agent that learns to complete laps.

### 2.1 Integration with Stable-Baselines3
- [ ] Wrap env with `VecEnv` or `DummyVecEnv` for vectorized rollouts
- [ ] Configure `PPO` with MLP policy (default `MlpPolicy`)
- [ ] Basic training loop with `model.learn()`

### 2.2 Hyperparameter Tuning
- [ ] **Discount factor (gamma)**: 0.99 typical; lower for shorter episodes
- [ ] **Reward weights**: lap progress vs. off-track penalty vs. speed
- [ ] **Learning rate, batch size, n_steps**: SB3 defaults are a good start
- [ ] **Normalization**: enable `normalize_advantage` and observation normalization

### 2.3 Training Infrastructure
- [ ] TensorBoard logging via `model.learn(..., callback=[TensorboardCallback])`
- [ ] Custom callbacks: log lap times, off-track count, progress
- [ ] Checkpoint saving: `model.save()`, periodic saves

### 2.4 Validation
- [ ] Policy learns to stay on track on a simple oval
- [ ] Metrics improve over training (reward, episode length, lap completion)

---

## Phase 3: Curriculum Learning

**Goal:** Train on progressively harder tracks and speed targets.

### 3.1 Track Difficulty
- [ ] **Easy**: wide track, gentle curves, short lap
- [ ] **Medium**: narrower track, sharper curves
- [ ] **Hard**: narrow, hairpins, chicane
- [ ] Parameterize: `track_width`, `curvature`, `lap_length`

### 3.2 Curriculum Logic
- [ ] Track registry: list of `(track_id, difficulty)` or procedural params
- [ ] **Strategy A**: switch track when success rate > threshold
- [ ] **Strategy B**: schedule (e.g., every N steps)
- [ ] **Strategy C**: mix episodes from easy/medium/hard
- [ ] Implement `CurriculumWrapper` that changes env params on `reset`

### 3.3 Speed Targets
- [ ] Optional: curriculum over target speed (start slow, increase)
- [ ] Reward shaping: bonus for exceeding speed threshold (only when on track)

### 3.4 Validation
- [ ] Curriculum improves final performance vs. training only on hard track
- [ ] Log which track/speed each episode used

---

## Phase 4: Rule-Based Baseline

**Goal:** A simple controller to compare against the PPO agent.

### 4.1 Baseline Design
- [ ] Use track-relative features: heading error, distance to centerline
- [ ] **Steering**: proportional to heading error + cross-track error
- [ ] **Throttle**: constant or speed-dependent (slow in curves)
- [ ] No learning; deterministic policy

### 4.2 Implementation
- [ ] `RuleBasedAgent` class: `act(obs) -> action`
- [ ] Compatible with same env; run N evaluation episodes
- [ ] Log: lap time, off-track count, success rate

### 4.3 Comparison
- [ ] Script to run PPO policy vs. rule-based on same tracks
- [ ] Report: mean lap time, std, off-track events, lap completion %

---

## Phase 5: Procedural Tracks & Robustness

**Goal:** Evaluate on many track variants; ensure policy generalizes.

### 5.1 Procedural Track Generation
- [ ] **Method A**: Random waypoints + smooth spline (e.g., cubic)
- [ ] **Method B**: Random segments (straight, left, right) with configurable curvature
- [ ] **Method C**: Noise on a base track shape
- [ ] Generate N variants for train vs. validation vs. test

### 5.2 Evaluation Protocol
- [ ] **Train set**: subset of procedural tracks (or curriculum)
- [ ] **Validation set**: held-out tracks; used for early stopping / checkpoint selection
- [ ] **Test set**: final evaluation; never seen during training
- [ ] Report mean ± std across tracks

### 5.3 Checkpointing by Validation Lap Time
- [ ] Custom callback: evaluate on validation tracks every K steps
- [ ] Save best model by mean validation lap time (or success rate)
- [ ] Track `best_mean_lap_time`, `best_model_path`

---

## Phase 6: Hyperparameter Sweeps & Final Polish

**Goal:** Systematic tuning and reproducible results.

### 6.1 Sweep Design
- [ ] Key params: gamma, reward weights, learning rate, curriculum schedule
- [ ] Use `Optuna`, `wandb sweep`, or simple grid/random search
- [ ] Log all runs to TensorBoard (or W&B) with tags

### 6.2 Reproducibility
- [ ] Seed env, policy, numpy, torch
- [ ] Save full config (reward weights, env params) with checkpoints
- [ ] `README.md` with install, train, eval commands

### 6.3 Resume Description Alignment
- [ ] **Fully simulated 2D racing environment** ✓
- [ ] **Custom physics (steering/throttle, track boundaries, state)** ✓
- [ ] **Reward shaping (lap progress, on-track)** ✓
- [ ] **PPO + MLP, discount/reward tuning** ✓
- [ ] **Curriculum learning** ✓
- [ ] **Rule-based baseline comparison** ✓
- [ ] **Procedural tracks, hyperparameter sweeps** ✓
- [ ] **TensorBoard + checkpoint by validation lap time** ✓

---

## Suggested Project Structure

```
reinforcement-learning-racecar/
├── README.md
├── requirements.txt
├── ROADMAP.md
├── src/
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── racecar_env.py      # Main Gymnasium env
│   │   ├── physics.py          # Steering, throttle, kinematics
│   │   ├── track.py            # Track representation, boundaries
│   │   └── rewards.py          # Reward functions
│   ├── curriculum/
│   │   ├── __init__.py
│   │   └── curriculum_wrapper.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── rule_based.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py         # Eval script, metrics
│   └── training/
│       ├── __init__.py
│       ├── train.py            # Main training script
│       └── callbacks.py        # TensorBoard, checkpointing
├── tracks/                     # Track definitions (JSON/YAML)
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── sweep.py
├── runs/                       # TensorBoard logs
└── checkpoints/                # Saved models
```

---

## Recommended Order of Implementation

1. **Phase 1** (env + physics) — foundation for everything
2. **Phase 2** (PPO training) — prove the loop works
3. **Phase 4** (rule-based baseline) — quick to add; useful baseline early
4. **Phase 3** (curriculum) — improves training quality
5. **Phase 5** (procedural tracks + robustness) — generalization
6. **Phase 6** (sweeps + polish) — final tuning and resume alignment

---

## Time Estimates (Rough)

| Phase | Estimate |
|-------|----------|
| 1. Core environment | 1–2 weeks |
| 2. PPO pipeline | 3–5 days |
| 3. Curriculum | 3–5 days |
| 4. Rule-based baseline | 1–2 days |
| 5. Procedural + eval | 1 week |
| 6. Sweeps + polish | 3–5 days |

Total: ~4–6 weeks part-time, or ~2 weeks full-time.

---

## Key Technical Decisions to Make Early

1. **Action space**: Continuous `Box(steering, throttle)` vs. discrete (e.g., 5×5 grid)
2. **Track format**: Pre-defined waypoints vs. fully procedural
3. **Progress metric**: Arc-length along centerline vs. checkpoint-based
4. **Curriculum**: Automatic (success-based) vs. scheduled vs. mixed

---

## References

- [Gymnasium docs](https://gymnasium.farama.org/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/stable/modules/ppo.html)
- Bicycle model: [Ackermann steering geometry](https://en.wikipedia.org/wiki/Ackermann_steering_geometry)
