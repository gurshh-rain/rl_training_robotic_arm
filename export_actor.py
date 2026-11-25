import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "ppo_attachment_arm_direct"
DEFAULT_OUTPUT_PATH = ROOT / "robot_recovery" / "rl_actor_weights.h"
EXPECTED_DIMS = (13, 32, 32, 2)


def format_float(value):
    return f"{float(value):.9e}f"


def format_vector(name, values):
    body = ", ".join(format_float(value) for value in values)
    return f"const float {name}[{len(values)}] = {{\n    {body}\n}};"


def format_matrix(name, values):
    rows = [
        "    {" + ", ".join(format_float(value) for value in row) + "}"
        for row in values
    ]
    return (
        f"const float {name}[{values.shape[0]}][{values.shape[1]}] = {{\n"
        + ",\n".join(rows)
        + "\n};"
    )


def extract_actor(model):
    metadata = getattr(model, "deployment_metadata", {})
    if metadata.get("action_semantics") != "direct_joint_targets_v1":
        raise ValueError("Refusing to export a residual or unlabelled checkpoint as direct control")
    if metadata.get("observation_version") != "gravity_gyro_targets_progress_side_v1":
        raise ValueError("Unsupported observation contract")
    if metadata.get("side_order") != ["+x", "-x", "+z", "-z"]:
        raise ValueError("Unsupported side encoding")
    if (metadata.get("control_period_seconds"), metadata.get("episode_seconds"),
        metadata.get("target_speed_radians_per_second")) != (0.02, 12.5, 2.0):
        raise ValueError("Unsupported controller timing")
    if model.policy.squash_output or not np.all(model.action_space.low == -1.0) or not np.all(model.action_space.high == 1.0):
        raise ValueError("Expected unsquashed, clipped [-1, 1] actions")
    modules = list(model.policy.mlp_extractor.policy_net)
    if [type(module) for module in modules] != [torch.nn.Linear, torch.nn.Tanh, torch.nn.Linear, torch.nn.Tanh]:
        raise ValueError("Expected Linear/Tanh/Linear/Tanh actor")
    hidden_layers = [
        module
        for module in model.policy.mlp_extractor.policy_net
        if isinstance(module, torch.nn.Linear)
    ]
    output_layer = model.policy.action_net
    if len(hidden_layers) != 2 or not isinstance(output_layer, torch.nn.Linear):
        raise ValueError("Expected two hidden Linear layers and one Linear action layer")

    layer1, layer2 = hidden_layers
    dimensions = (
        layer1.in_features,
        layer1.out_features,
        layer2.out_features,
        output_layer.out_features,
    )
    if dimensions != EXPECTED_DIMS or layer2.in_features != layer1.out_features or output_layer.in_features != layer2.out_features:
        raise ValueError(f"Expected actor dimensions {EXPECTED_DIMS}, found {dimensions}")

    return {
        "layer1Weights": layer1.weight.detach().cpu().numpy().T.astype(np.float32),
        "layer1Bias": layer1.bias.detach().cpu().numpy().astype(np.float32),
        "layer2Weights": layer2.weight.detach().cpu().numpy().T.astype(np.float32),
        "layer2Bias": layer2.bias.detach().cpu().numpy().astype(np.float32),
        "outputWeights": output_layer.weight.detach().cpu().numpy().T.astype(np.float32),
        "outputBias": output_layer.bias.detach().cpu().numpy().astype(np.float32),
    }


def actor_forward(observations, actor):
    hidden1 = np.tanh(observations @ actor["layer1Weights"] + actor["layer1Bias"])
    hidden2 = np.tanh(hidden1 @ actor["layer2Weights"] + actor["layer2Bias"])
    return np.clip(hidden2 @ actor["outputWeights"] + actor["outputBias"], -1.0, 1.0)


def validation_observations(count=256):
    rng = np.random.default_rng(0)
    observations = np.zeros((count, EXPECTED_DIMS[0]), dtype=np.float32)
    gravity = rng.normal(size=(count, 3)).astype(np.float32)
    observations[:, :3] = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    observations[:, 3:6] = rng.normal(0.0, 2.0, size=(count, 3))
    observations[:, 6:8] = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=(count, 2))
    observations[:, 8] = rng.uniform(0.0, 1.0, size=count)
    side_indices = rng.integers(0, 4, size=count)
    observations[np.arange(count), 9 + side_indices] = 1.0
    return observations


def validate(model, actor):
    observations = validation_observations()
    stable_baselines_actions, _ = model.predict(observations, deterministic=True)
    exported_actions = actor_forward(observations, actor)
    maximum_error = float(np.max(np.abs(stable_baselines_actions - exported_actions)))
    if maximum_error > 1e-5:
        raise ValueError(f"Export validation failed with maximum error {maximum_error}")
    return maximum_error


def write_header(path, actor, simulation_validated, checkpoint_hash):
    sections = [
        "#pragma once",
        "#include <stddef.h>",
        'constexpr bool actorDirectControl = true;',
        f'constexpr bool actorSimulationValidated = {str(simulation_validated).lower()};',
        'constexpr float actorControlPeriodSeconds = 0.02f;',
        'constexpr float actorEpisodeSeconds = 12.5f;',
        'constexpr float actorTargetSpeed = 2.0f;',
        f'const char actorCheckpointSha256[] = "{checkpoint_hash}";',
        "const size_t actorInputSize = 13;",
        "const size_t actorHidden1Size = 32;",
        "const size_t actorHidden2Size = 32;",
        "const size_t actorOutputSize = 2;",
        format_matrix("layer1Weights", actor["layer1Weights"]),
        format_vector("layer1Bias", actor["layer1Bias"]),
        format_matrix("layer2Weights", actor["layer2Weights"]),
        format_vector("layer2Bias", actor["layer2Bias"]),
        format_matrix("outputWeights", actor["outputWeights"]),
        format_vector("outputBias", actor["outputBias"]),
    ]
    path.write_text("\n\n".join(sections) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model)
    if model.observation_space.shape != (EXPECTED_DIMS[0],):
        raise ValueError(f"Expected observation shape {(EXPECTED_DIMS[0],)}, found {model.observation_space.shape}")
    if model.action_space.shape != (EXPECTED_DIMS[-1],):
        raise ValueError(f"Expected action shape {(EXPECTED_DIMS[-1],)}, found {model.action_space.shape}")

    actor = extract_actor(model)
    maximum_error = validate(model, actor)
    simulation_validated = bool(getattr(model, "simulation_validated", False))
    if not simulation_validated and not args.allow_unvalidated:
        raise ValueError("Checkpoint has not passed held-out recovery evaluation; use --allow-unvalidated only for diagnostic export")
    checkpoint_path = args.model if args.model.suffix == ".zip" else args.model.with_suffix(".zip")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    write_header(args.output, actor, simulation_validated, checkpoint_hash)
    print(f"Direct-control export; simulation validation passed: {simulation_validated}")

    parameter_count = sum(values.size for values in actor.values())
    print(f"Exported {parameter_count} actor parameters to {args.output}")
    print(f"Float32 storage: {parameter_count * 4} bytes")
    print(f"Maximum validation error: {maximum_error:.3e}")


if __name__ == "__main__":
    main()
