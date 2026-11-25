from types import SimpleNamespace

from export_actor import actor_forward, extract_actor, validation_observations, write_header
import numpy as np
import pytest
from stable_baselines3 import PPO

from pushup_env import ArmPushupEnv
from train_rl import POLICY_METADATA, require_direct_policy


def test_legacy_checkpoint_cannot_be_exported_as_direct_control():
    model = SimpleNamespace()
    with pytest.raises(ValueError, match="residual or unlabelled"):
        extract_actor(model)
    with pytest.raises(ValueError, match="direct-control"):
        require_direct_policy(model)


def test_actor_export_contract_and_numerical_parity(tmp_path):
    env = ArmPushupEnv(domain_randomization=False)
    try:
        model = PPO(
            "MlpPolicy", env, n_steps=32, batch_size=32,
            policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [64, 64]}},
            device="cpu", seed=0,
        )
        model.deployment_metadata = POLICY_METADATA.copy()
        require_direct_policy(model)
        actor = extract_actor(model)
        observations = validation_observations()
        expected, _ = model.predict(observations, deterministic=True)
        np.testing.assert_allclose(actor_forward(observations, actor), expected, atol=1e-6)
        header = tmp_path / "weights.h"
        write_header(header, actor, False, "0" * 64)
        text = header.read_text()
        assert "actorDirectControl = true" in text
        assert "actorSimulationValidated = false" in text
        assert "layer1Weights[13][32]" in text
        assert "outputWeights[32][2]" in text
        model.deployment_metadata["action_semantics"] = "residual"
        with pytest.raises(ValueError):
            extract_actor(model)
    finally:
        env.close()
