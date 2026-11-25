import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from pushup_env import ArmPushupEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 3_000_000
# Mean |target step| / slew limit while stably upright. Bang-bang dithering is ~1.0.
MAX_UPRIGHT_TARGET_RATE = 0.25
# Exploration noise must stay well inside the [-1, 1] action range. An entropy
# bonus previously grew std to ~3, so nearly every sampled action clipped to
# +/-1 and the policy learned bang-bang targets that chatter when upright.
ENT_COEF = 0.0
INITIAL_ACTION_STD = 0.4
MODEL_PATH = Path(__file__).resolve().parent / "ppo_attachment_arm_direct"
POLICY_METADATA = {
    "action_semantics": ArmPushupEnv.POLICY_CONTRACT,
    "observation_version": "gravity_gyro_targets_progress_side_v1",
    "observation_size": 13,
    "control_period_seconds": 0.02,
    "episode_seconds": 12.5,
    "joint_limit_radians": float(np.pi / 2.0),
    "target_speed_radians_per_second": 2.0,
    "side_order": list(ArmPushupEnv.FALL_SIDES),
}


def require_direct_policy(model):
    if getattr(model, "deployment_metadata", None) != POLICY_METADATA:
        raise ValueError("This checkpoint does not declare the direct-control observation/action contract")


def evaluate(model, episodes_per_configuration=5, domain_randomization=True, seed_start=10000, bearings=0):
    require_direct_policy(model)
    report = {}
    env = ArmPushupEnv(domain_randomization=domain_randomization)
    try:
        starts = ({f"{angle:g}deg": {"fall_bearing_deg": angle}
                   for angle in np.linspace(0, 360, bearings, endpoint=False)}
                  if bearings else {side: {"fall_side": side} for side in env.FALL_SIDES})
        for label, options in starts.items():
            outcomes = []
            unsupported = 0
            for seed in range(seed_start, seed_start + episodes_per_configuration):
                obs, _ = env.reset(seed=seed, options=options)
                if bearings and (max(abs(obs[0]), abs(obs[2])) < 0.8 or obs[1] > 0.85):
                    # Not a supported side after settling. Don't credit a drop
                    # that settled upright or use a side the hardware can't see.
                    unsupported += 1
                    outcomes.append((False, False, 0.0, 0.0, np.nan))
                    continue
                reached = False
                poses = []
                upright_rates = []
                for _ in range(env.MAX_STEPS):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    reached |= bool(info["is_success"])
                    poses.append(info["pose_score"])
                    if info["stable_upright"]:
                        upright_rates.append(info["target_rate"])
                    if terminated or truncated:
                        break
                outcomes.append((
                    bool(info["is_success"]) and not terminated,
                    reached,
                    float(np.mean(poses[-env.STABLE_SUCCESS_STEPS:])),
                    info["upright_steps"] * env.dt,
                    float(np.mean(upright_rates)) if upright_rates else np.nan,
                ))
            values = np.asarray(outcomes, dtype=np.float64)
            report[label] = {
                "episodes": episodes_per_configuration,
                "final_success_rate": float(values[:, 0].mean()),
                "ever_success_rate": float(values[:, 1].mean()),
                "mean_tail_pose_score": float(values[:, 2].mean()),
                "minimum_final_hold_seconds": float(values[:, 3].min()),
                "upright_target_rate": (
                    float(np.nanmean(values[:, 4])) if np.isfinite(values[:, 4]).any() else None
                ),
            }
            if bearings:
                report[label]["non_side_starts"] = unsupported
    finally:
        env.close()
    return report


def evaluate_modes(model, episodes=5, seed_start=10000, bearings=0):
    return {
        "nominal": evaluate(model, episodes, False, seed_start, bearings),
        "randomized": evaluate(model, episodes, True, seed_start, bearings),
    }


def report_jitter(report):
    rates = [row["upright_target_rate"] for mode in report.values() for row in mode.values()]
    return max((rate for rate in rates if rate is not None), default=1.0)


def report_score(report):
    rows = [row for mode in report.values() for row in mode.values()]
    return (
        float(report_jitter(report) <= MAX_UPRIGHT_TARGET_RATE),
        min(row["final_success_rate"] for row in rows),
        float(np.mean([row["final_success_rate"] for row in rows])),
        float(np.mean([row["mean_tail_pose_score"] for row in rows])),
    )


class BestDirectPolicy(BaseCallback):
    def __init__(self, path, every=250000):
        super().__init__()
        self.path = path
        self.every = every
        self.last_evaluation = 0
        self.best_score = (-1.0, -1.0, -1.0, -1.0)

    def evaluate_and_save(self):
        report = evaluate_modes(self.model, episodes=3)
        score = report_score(report)
        print(f"Evaluation at {self.num_timesteps}: {json.dumps(report)}", flush=True)
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.path)
            print(f"Saved direct-policy candidate: {self.path}.zip, score={score}", flush=True)
        self.last_evaluation = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self.last_evaluation >= self.every:
            self.evaluate_and_save()
        return True

    def _on_training_end(self):
        self.evaluate_and_save()


def train(total_timesteps=TOTAL_TIMESTEPS, resume=None, seed=0, eval_every=250000):
    env = DummyVecEnv([
        lambda: ArmPushupEnv(upright_reset_probability=0.15)
        for _ in range(N_ENVS)
    ])
    try:
        if resume:
            model = PPO.load(resume, env=env, device="cpu")
            require_direct_policy(model)
            model.ent_coef = ENT_COEF
            with torch.no_grad():
                model.policy.log_std.fill_(np.log(INITIAL_ACTION_STD))
        else:
            model = PPO(
                "MlpPolicy", env, learning_rate=3e-4, n_steps=512,
                batch_size=256, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                ent_coef=ENT_COEF, policy_kwargs={
                    "net_arch": {"pi": [32, 32], "vf": [64, 64]},
                    "log_std_init": float(np.log(INITIAL_ACTION_STD)),
                },
                seed=seed, verbose=0, device="cpu",
            )
            model.deployment_metadata = POLICY_METADATA.copy()
        model.simulation_validated = False
        callback = BestDirectPolicy(MODEL_PATH, every=eval_every)
        print("Training direct joint targets from reward only: no references, imitation, or forced homing.", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=not bool(resume))
        model.save(MODEL_PATH.with_name(MODEL_PATH.name + "_latest"))
    finally:
        env.close()

    best = PPO.load(MODEL_PATH, device="cpu")
    report = evaluate_modes(best, episodes=10, seed_start=20000)
    best.simulation_validated = (
        all(row["final_success_rate"] >= 0.9 for row in report["nominal"].values()) and
        all(row["final_success_rate"] >= 0.8 for row in report["randomized"].values()) and
        report_jitter(report) <= MAX_UPRIGHT_TARGET_RATE
    )
    best.validation_report = report
    best.save(MODEL_PATH)
    print(f"Held-out evaluation: {json.dumps(report, indent=2)}", flush=True)
    print(f"Simulation validation passed: {best.simulation_validated}", flush=True)
    print(f"Saved {MODEL_PATH}.zip (old residual checkpoints untouched)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250000)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--bearings", type=int, default=0, help="evaluate N evenly spaced fall bearings instead of four named sides")
    args = parser.parse_args()
    if args.bearings < 0 or (args.bearings and not args.evaluate):
        parser.error("--bearings requires --evaluate and a non-negative count")
    if args.steps <= 0 or args.eval_every <= 0 or args.episodes <= 0:
        parser.error("steps, eval-every, and episodes must be positive")
    if args.evaluate:
        policy = PPO.load(args.resume or MODEL_PATH, device="cpu")
        print(json.dumps(evaluate_modes(policy, args.episodes, seed_start=30000, bearings=args.bearings), indent=2))
    else:
        train(args.steps, args.resume, args.seed, args.eval_every)
