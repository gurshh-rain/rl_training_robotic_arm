import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from pushup_env import ArmPushupEnv


MODEL_PATH = Path(__file__).resolve().parent / "ppo_attachment_arm"
FALLEN_DISPLAY_SECONDS = 2.0


def reset_env(env, fall_side):
    options = None if fall_side == "random" else {"fall_side": fall_side}
    return env.reset(options=options)


def display_fallen_pose(viewer):
    deadline = time.time() + FALLEN_DISPLAY_SECONDS
    while viewer.is_running() and time.time() < deadline:
        viewer.sync()
        time.sleep(0.01)


def restart_recovery(env):
    rotation = env.data.xmat[env.base_body_id].reshape(3, 3)
    gravity_in_base = rotation.T @ [0.0, 0.0, 1.0]
    if abs(gravity_in_base[2]) >= abs(gravity_in_base[0]):
        env.fall_side = "+x" if gravity_in_base[2] >= 0.0 else "-x"
    else:
        env.fall_side = "+z" if gravity_in_base[0] < 0.0 else "-z"
    env.step_count = 0
    env.upright_steps = 0
    env.previous_upright = env._base_upright()
    return env._get_obs()


def test_trained_model(fall_side="random"):
    env = ArmPushupEnv()
    model = PPO.load(MODEL_PATH, env=env)
    obs, reset_info = reset_env(env, fall_side)

    print(f"Loaded {MODEL_PATH}.zip")
    print("The arm will recover automatically. Once upright, push it with the MuJoCo viewer controls.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print(
            f"Starting on side {reset_info['fall_side']} "
            f"after {FALLEN_DISPLAY_SECONDS:.0f} seconds..."
        )
        display_fallen_pose(viewer)
        was_upright = False

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, info = env.step(action)
            viewer.sync()
            time.sleep(env.dt)

            if info["is_success"] and not was_upright:
                was_upright = True
                print("Upright. Push the model over to test recovery.")
            elif was_upright and info["upright"] < 0.55:
                was_upright = False
                obs = restart_recovery(env)
                print(f"Fall detected on side {env.fall_side}. Restarting recovery.")

            if terminated:
                obs, reset_info = reset_env(env, fall_side)
                was_upright = False
                print(f"Resetting on side {reset_info['fall_side']} after an unstable fall...")
                display_fallen_pose(viewer)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fall-side",
        choices=(*ArmPushupEnv.FALL_SIDES, "random"),
        default="random",
    )
    test_trained_model(parser.parse_args().fall_side)
