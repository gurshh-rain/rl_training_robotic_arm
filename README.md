# RL Robotic Arm - Push-up Task

This project implements a reinforcement learning agent designed to train a robotic arm to recover from fallen positions and return to an upright posture (a "push-up" motion).

## Checking a real robot

The actor assumes a base-mounted IMU. With servo power disconnected and the
ESP32 disarmed, inspect the 115200-baud status while holding the base still.
Upright should give `Up` near `(0,-1,0)` and `Side=FLAT`. Rotating the base
+90 degrees about its model X axis should give `(0,0,1)` / `+x`; about its
model Z axis it should give `(-1,0,0)` / `+z`. Opposite rotations reverse
the signs. If these disagree, correct `IMU_TO_BASE` (a proper rotation) in
the sketch before attempting recovery. Gyro calibration does not set mounting
orientation. `PolicyUp` is the gyro-assisted up direction used by the actor.

Build the complete `robot_recovery` sketch folder (including both headers).
On first attachment the center command can shift the base. The firmware waits
for another 500 ms of stationary side readings before selecting `EpisodeSide`.
If the powered base cannot settle on a supported pose within 3 s of entering
that wait, `POWERED_SETTLE_TIMEOUT` detaches the servos.

If the sensors agree, distinguish a gate/fault from an unsuccessful motion:
check `Gate`, `EpisodeSide`, `TargetsDeg`, and `PWMus`. PWM reports the
command, not the shaft angle. Verify horn zero, joint direction, available
travel, common ground and supply voltage under load rather than widening
pulse limits. No motion is authorized by a passing simulation test.

To exercise intermediate drop directions with the existing actor:
`mamba run -n ros_humble python train_rl.py --evaluate --episodes 5 --bearings 16`.
This varies the **local fall axis**, lets the robot settle, then selects the
actor side from the observed resting pose. `non_side_starts` count as misses.
This tests initial bearings, not every possible resting pose: the firmware
still waits on unrecognized tilts and does not attempt inverted recovery.

## Overview

The robotic arm is simulated in a MuJoCo environment. The goal is to learn a control policy that can handle various initial fall orientations and move the arm back to a stable, upright position.

## Key Components

### 1. Environment (`pushup_env.py`)
- **Simulation:** Built using MuJoCo and wrapped in a Gymnasium environment.
- **State Space:** Base up direction, body-frame gyro, commanded joint targets, normalized episode time, and a four-way starting-side label.
- **Action Space:** Continuous control of the robotic arm's servos.
- **Reward Function:** A multi-objective reward including:
    - **Upright Score:** Reward for maintaining a vertical base.
    - **Progress:** Bonus for increasing the upright score.
    - **Contact Reward:** Encourages the arm's tip to make contact with the floor during recovery.
    - **Pose Reward:** Incentivizes reaching a target "home" pose once upright.
    - **Penalties:** Penalties for excessive height, high-velocity movements, and large action changes.
- **Fall Configurations:** Supports multiple fall sides (`+x`, `-x`, `+z`, `-z`) to ensure robust recovery.

### 2. Training (`train_rl.py`)
- **Algorithm:** Proximal Policy Optimization (PPO) from `stable-baselines3`.
- **Training Pipeline:**
    1. **Training:** PPO learns direct joint targets from reward across headless environments, without scripted motions or imitation.
    2. **Evaluation:** Checkpoints are compared using sustained end-of-episode recovery on all four sides.

### 3. Visualization (`pushup_upright.py`)
- A script to load a trained model and visualize the arm's recovery in the MuJoCo viewer.
- Supports testing specific fall sides or random initializations.

## Project Structure

- `train_rl.py`: Main training script.
- `pushup_env.py`: MuJoCo environment definition.
- `pushup_upright.py`: Visualization script for trained models.
- `arm_control.py`: Basic arm control utilities.
- `models/`: Contains robot models (URDF, XML, and STL files).
- `rl_robotic_arm_v1/`: ROS2 package source.
- `package.xml`, `setup.py`, `setup.cfg`: ROS2 package configuration files.

## How to Use

### Training
To start the training process:
```bash
python train_rl.py
```

### Visualization
To visualize the trained model:
```bash
python pushup_upright.py --fall-side random
```
You can replace `random` with one of the fall sides: `+x`, `-x`, `+z`, `-z`.
