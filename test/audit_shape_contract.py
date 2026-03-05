"""Independent audit: verify shape contract with different parameters."""

import sys

sys.path.insert(0, "/home/rec/server/rsl_rl")

import torch
from tensordict import TensorDict
from rsl_rl.storage.rollout_storage import RolloutStorage


def audit_with_different_params():
    """Test with different parameters to verify robustness."""
    print("=" * 80)
    print("INDEPENDENT AUDIT: Testing with different parameters")
    print("=" * 80)

    test_configs = [
        {"num_envs": 2, "num_transitions": 4, "obs_dim": 5, "action_dim": 2},
        {"num_envs": 8, "num_transitions": 16, "obs_dim": 20, "action_dim": 6},
        {"num_envs": 16, "num_transitions": 32, "obs_dim": 48, "action_dim": 12},
    ]

    all_passed = True

    for idx, params in enumerate(test_configs):
        print(f"\n{'=' * 80}")
        print(f"Test Config {idx + 1}: {params}")
        print(f"{'=' * 80}")

        obs_example = TensorDict(
            {"policy": torch.zeros(params["num_envs"], params["obs_dim"])},
            batch_size=[params["num_envs"]],
            device="cpu",
        )

        # Create storage
        storage = RolloutStorage(
            training_type="rl",
            num_envs=params["num_envs"],
            num_transitions_per_env=params["num_transitions"],
            obs=obs_example,
            actions_shape=(params["action_dim"],),
            device="cpu",
        )

        # Fill storage
        for step in range(params["num_transitions"]):
            transition = RolloutStorage.Transition()
            transition.observations = TensorDict(
                {"policy": torch.randn(params["num_envs"], params["obs_dim"])},
                batch_size=[params["num_envs"]],
                device="cpu",
            )
            transition.actions = torch.randn(params["num_envs"], params["action_dim"])
            transition.rewards = torch.randn(params["num_envs"])
            transition.dones = torch.zeros(params["num_envs"], dtype=torch.uint8)
            transition.values = torch.randn(params["num_envs"], 1)
            transition.actions_log_prob = torch.randn(params["num_envs"], 1)
            transition.action_mean = torch.randn(params["num_envs"], params["action_dim"])
            transition.action_sigma = torch.abs(torch.randn(params["num_envs"], params["action_dim"]))
            storage.add_transition(transition)

        # Test Feedforward
        ff_gen = storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
        ff_batch = next(ff_gen)
        ff_obs = ff_batch[0]["policy"] if isinstance(ff_batch[0], TensorDict) else ff_batch[0]

        expected_ff_batch = (params["num_transitions"] * params["num_envs"]) // 2
        print(f"\nFeedforward:")
        print(f"  Expected: ({expected_ff_batch}, {params['obs_dim']})")
        print(f"  Actual:   {ff_obs.shape}")

        if ff_obs.shape != (expected_ff_batch, params["obs_dim"]):
            print(f"  ❌ FAILED")
            all_passed = False
        else:
            print(f"  ✅ PASSED")

        # Verify dimensions
        if len(ff_obs.shape) != 2:
            print(f"  ❌ FAILED: Expected 2D, got {len(ff_obs.shape)}D")
            all_passed = False

    print(f"\n{'=' * 80}")
    if all_passed:
        print("✅ ALL AUDIT TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("❌ SOME AUDIT TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(audit_with_different_params())
