# hyperleg_rl

Bipedal **HyperLeg** (14-DoF) RL locomotion training/evaluation pipeline on Isaac Lab + rsl_rl PPO. Targets reproduction of the IROS 2025 / ICCAS 2026 *active toe* paper.

## Robot

- 14-DoF biped: 3-DoF hip + 1-DoF knee + 2-DoF ankle + **1-DoF active toe** per leg
- Coupled belt-linkage actuators (CA Jacobian) with 4-quadrant motor torque-speed clip
- Toe-ablation variant: 12 DoF (drops the TO joint)

## Tasks (gym IDs)

| ID | DoF | Purpose |
|---|---|---|
| `HyperLeg-Locomotion-v0`            | 14 | toe-equipped training (rough terrain) |
| `HyperLeg-Locomotion-Play-v0`       | 14 | toe-equipped playback (flat) |
| `HyperLeg-Locomotion-WoToe-v0`      | 12 | toe-ablation training |
| `HyperLeg-Locomotion-WoToe-Play-v0` | 12 | toe-ablation playback |

## Quick start

```bash
# Train (3000 iter, ~hours/run on one GPU)
./isaaclab.sh -p hyperleg_rl/scripts/rsl_rl/train.py \
    --task HyperLeg-Locomotion-v0 --headless --num_envs 4096

# Play with goal-tracking + per-trial power/GRF CSV logging
./isaaclab.sh -p hyperleg_rl/scripts/rsl_rl/play_goto_x.py \
    --task HyperLeg-Locomotion-Play-v0 \
    --checkpoint logs/rsl_rl/hyperleg_locomotion/<run>/model_2999.pt \
    --num_envs 1 --goal_x 20.0 --speed 1.33 --real-time --log_csv
```

## ICCAS Table II reproduction

`play_goto_x.py --log_csv` runs 10 walk-to-goal trials at 1.33 m/s and writes a per-step CSV per trial:

- Path: `logs/ICCAS/{toe_equipped|toe_ablation}_<YYYY-MM-DD_HH-MM-SS>/trial_NN.csv`
- Columns: `t_ep_s, joule_heating, mech_power, total_power, base_vx_w, l_heel_grf, r_heel_grf`
- Steady window: episode time `[2, 14]` s (skip startup transient)

Offline mean across 10 CSVs reproduces the Table II row. Power model (paper-faithful):

```
joule = Cp · τ_m²                      # I²R proxy (raw motor torque)
p     = τ_m · ω_m
mech  = where(p<0, p·η_regen, p/η_out) # η_regen=0.8, η_out=0.9
CoT   = mean(P_total) / (m·g·mean(base_vx_w))
```

## Live telemetry (`--telemetry`)

Three-tab ImPlot window for env-0:
- **Motors**: joint torque + motor heat EMA, per-role L/R overlay
- **Power**: per-group (Hip/Knee/Ankle/Toe) power + Joule/Mech/Total breakdown
- **GRF**: heel ground-reaction force, peak tracker, time series

## Layout

```
scripts/rsl_rl/                    train.py / play.py / play_goto_x.py
source/hyperleg_rl/
  actuators/coupled_leg.py         CA Jacobian + 4-quadrant motor clip
  assets/hyperleg.py               HYPERLEG_CFG / HYPERLEG_WO_TOE_CFG
  sensors/                         ForceVectorContactSensor (debug-draw GRF arrows)
  tasks/.../hyperleg/
    hyperleg_env_cfg.py            train (rough) + play (flat) env cfgs
    mdp/rewards.py                 actuator_power_consumption, thermal_penalty, ...
    mdp/observations.py            cot_components, heel_grf_magnitude, motor_heat
  viz/
    motor_telemetry.py             live Motors/Power/GRF tabs
    power_logger.py                per-trial CSV (single GPU sync/step)
dvcc/                              dev notes (gitignored, see 03_commit_convention.md)
logs/                              outputs (gitignored)
```
