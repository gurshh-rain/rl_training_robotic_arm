import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pushup_env import ArmPushupEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
MODEL_PATH = Path(__file__).resolve().parent / "ppo_attachment_arm"


def collect_reference_data(episodes_per_configuration=2):
    observations = []
    actions = []
    env = ArmPushupEnv()

    for side in env.FALL_SIDES:
        for seed in range(episodes_per_configuration):
            obs, _ = env.reset(seed=seed, options={"fall_side": side})
            for _ in range(env.MAX_STEPS):
                action = np.zeros(2, dtype=np.float32)
                observations.append(obs)
                actions.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    env.close()
    return np.asarray(observations), np.asarray(actions)


def pretrain_from_reference(model, epochs=80, batch_size=512):
    observations, actions = collect_reference_data()
    observations = torch.as_tensor(observations, device=model.device)
    actions = torch.as_tensor(actions, device=model.device)
    learning_rates = [group["lr"] for group in model.policy.optimizer.param_groups]
    for group in model.policy.optimizer.param_groups:
        group["lr"] = 1e-3

    for _ in range(epochs):
        for indices in torch.randperm(len(observations), device=model.device).split(batch_size):
            distribution = model.policy.get_distribution(observations[indices])
            loss = F.mse_loss(distribution.distribution.mean, actions[indices])
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()

    for group, learning_rate in zip(model.policy.optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate


def evaluate(model, episodes_per_configuration=3):
    max_upright = []
    successes = 0
    env = ArmPushupEnv()

    for side in env.FALL_SIDES:
        for seed in range(episodes_per_configuration):
            obs, _ = env.reset(seed=seed, options={"fall_side": side})
            episode_max = 0.0
            episode_success = False

            for _ in range(env.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                episode_max = max(episode_max, info["upright"])
                episode_success |= info["is_success"]
                if terminated or truncated:
                    break

            max_upright.append(episode_max)
            successes += episode_success

    env.close()
    return float(np.mean(max_upright)), successes / len(max_upright)


def train():
    env = DummyVecEnv([ArmPushupEnv for _ in range(N_ENVS)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0,
        verbose=1,
    )

    print("Learning the verified get-up motion...")
    pretrain_from_reference(model)
    best_state = deepcopy(model.policy.state_dict())
    best_metrics = evaluate(model)

    print(f"Fine-tuning with PPO in {N_ENVS} headless environments...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    final_metrics = evaluate(model)
    if (final_metrics[1], final_metrics[0]) < (best_metrics[1], best_metrics[0]):
        model.policy.load_state_dict(best_state)
        final_metrics = best_metrics

    model.save(MODEL_PATH)
    env.close()

    mean_max_upright, success_rate = final_metrics
    print(f"Model saved to {MODEL_PATH}.zip")
    print(f"Mean maximum upright score: {mean_max_upright:.3f}")
    print(f"Get-up success rate: {success_rate:.0%}")
    print("Watch it with: python pushup_upright.py")


if __name__ == "__main__":
    train()
