# RL Robotic Arm - Push-up Task

This project implements a reinforcement learning agent designed to train a robotic arm to recover from fallen positions and return to an upright posture (a "push-up" motion).

## Overview

The robotic arm is simulated in a MuJoCo environment. The goal is to learn a control policy that can handle various initial fall orientations and move the arm back to a stable, upright position.

## Key Components

### 1. Environment (`pushup_env.py`)
- **Simulation:** Built using MuJoCo and wrapped in a Gymnasium environment.
- **State Space:** Includes base orientation (relative to gravity), base height, linear and angular velocities, joint positions and velocities, current servo targets, and normalized time.
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
    1. **Pre-training:** The model is first pre-trained using reference data (Behavioral Cloning style) to provide a baseline "get-up" motion.
    2. **Fine-tuning:** The model is then fine-tuned using PPO across multiple parallel headless environments to optimize the reward function.
    3. **Evaluation:** Periodic evaluation is performed to track success rates and mean upright scores.

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
