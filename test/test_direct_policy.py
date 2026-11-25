import numpy as np
import pytest

from pushup_env import ArmPushupEnv


@pytest.mark.parametrize("side", ArmPushupEnv.FALL_SIDES)
@pytest.mark.parametrize("step", [0, 99, 100, 300])
def test_policy_alone_selects_targets_at_every_phase(side, step):
    env = ArmPushupEnv(domain_randomization=False)
    try:
        env.reset(seed=4, options={"fall_side": side})
        env.step_count = step
        env.step(np.array([-0.75, 0.5], dtype=np.float32))
        np.testing.assert_allclose(env.servo_targets, [-0.04, 0.04], atol=1e-7)
    finally:
        env.close()


@pytest.mark.parametrize("side", ArmPushupEnv.FALL_SIDES)
def test_zero_policy_does_not_receive_scripted_recovery(side):
    env = ArmPushupEnv(domain_randomization=False)
    try:
        env.reset(seed=7, options={"fall_side": side})
        for _ in range(env.MAX_STEPS):
            _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            np.testing.assert_array_equal(env.servo_targets, [0.0, 0.0])
            if terminated or truncated:
                break
        assert not info["is_success"]
    finally:
        env.close()


def test_observation_matches_commanded_not_measured_joint_angles():
    env = ArmPushupEnv(domain_randomization=False)
    try:
        env.reset(seed=11, options={"fall_side": "+z"})
        env.servo_targets[:] = [0.25, -0.5]
        obs = env._get_obs()
        assert obs.shape == (13,)
        np.testing.assert_allclose(obs[6:8], [0.25, -0.5])
        np.testing.assert_array_equal(obs[9:], [0, 0, 1, 0])
    finally:
        env.close()
