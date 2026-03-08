"""Test script for group_dependent_std and action_std_groups functionality."""

import sys

sys.path.insert(0, "/home/rec/server/rsl_rl")

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic, ActorCriticCNN


def test_group_dependent_std_scalar():
    """Test group_dependent_std=False (all actions share one std)."""
    print("\n" + "=" * 80)
    print("TEST 1: group_dependent_std=False (scalar std)")
    print("=" * 80)

    obs = TensorDict({"low_dim": torch.randn(4, 10)}, batch_size=[4])
    obs_groups = {"policy": ["low_dim"], "critic": ["low_dim"]}

    model = ActorCritic(
        obs,
        obs_groups,
        num_actions=4,
        group_dependent_std=False,
    )

    print(f"std param shape: {model.std.shape} (expected: [1])")
    assert model.std.shape == torch.Size([1]), f"Expected [1], got {model.std.shape}"

    action = model.act(obs)
    print(f"action shape: {action.shape} (expected: [4, 4])")
    assert action.shape == torch.Size([4, 4])

    print("✅ PASSED: group_dependent_std=False works correctly")
    return True


def test_group_dependent_std_per_action():
    """Test group_dependent_std=True with no groups (each action has its own std)."""
    print("\n" + "=" * 80)
    print("TEST 2: group_dependent_std=True, action_std_groups=None (per-action std)")
    print("=" * 80)

    obs = TensorDict({"low_dim": torch.randn(4, 10)}, batch_size=[4])
    obs_groups = {"policy": ["low_dim"], "critic": ["low_dim"]}

    model = ActorCritic(
        obs,
        obs_groups,
        num_actions=4,
        group_dependent_std=True,
        action_std_groups=None,
    )

    print(f"std param shape: {model.std.shape} (expected: [4])")
    assert model.std.shape == torch.Size([4]), f"Expected [4], got {model.std.shape}"

    action = model.act(obs)
    print(f"action shape: {action.shape} (expected: [4, 4])")
    assert action.shape == torch.Size([4, 4])

    print("✅ PASSED: group_dependent_std=True with no groups works correctly")
    return True


def test_action_std_groups():
    """Test action_std_groups=[[0,2], [1,3]] (grouped std)."""
    print("\n" + "=" * 80)
    print("TEST 3: group_dependent_std=True, action_std_groups=[[0,2], [1,3]]")
    print("=" * 80)

    obs = TensorDict({"low_dim": torch.randn(4, 10)}, batch_size=[4])
    obs_groups = {"policy": ["low_dim"], "critic": ["low_dim"]}

    model = ActorCritic(
        obs,
        obs_groups,
        num_actions=4,
        group_dependent_std=True,
        action_std_groups=[[0, 2], [1, 3]],
    )

    print(f"std param shape: {model.std.shape} (expected: [2])")
    assert model.std.shape == torch.Size([2]), f"Expected [2], got {model.std.shape}"

    print(f"std_group_map: {model.std_group_map} (expected: [0, 1, 0, 1])")
    expected_map = torch.tensor([0, 1, 0, 1])
    assert torch.equal(model.std_group_map, expected_map), f"Expected {expected_map}, got {model.std_group_map}"

    action = model.act(obs)
    print(f"action shape: {action.shape} (expected: [4, 4])")
    assert action.shape == torch.Size([4, 4])

    print("✅ PASSED: action_std_groups works correctly")
    return True


def test_std_sharing_verification():
    """Verify that actions in the same group share the same std."""
    print("\n" + "=" * 80)
    print("TEST 4: Verify std sharing within groups")
    print("=" * 80)

    obs = TensorDict({"low_dim": torch.randn(100, 10)}, batch_size=[100])
    obs_groups = {"policy": ["low_dim"], "critic": ["low_dim"]}

    model = ActorCritic(
        obs,
        obs_groups,
        num_actions=4,
        group_dependent_std=True,
        action_std_groups=[[0, 2], [1, 3]],
        noise_std_type="scalar",
    )

    with torch.no_grad():
        model.std[0] = 0.5
        model.std[1] = 1.5

    model.eval()
    with torch.no_grad():
        model._update_distribution(model.get_actor_obs(obs))

    std = model.distribution.stddev
    print(f"std shape: {std.shape}")
    print(f"std[0, 0] (action 0, group 0): {std[0, 0].item():.6f}")
    print(f"std[0, 1] (action 1, group 1): {std[0, 1].item():.6f}")
    print(f"std[0, 2] (action 2, group 0): {std[0, 2].item():.6f}")
    print(f"std[0, 3] (action 3, group 1): {std[0, 3].item():.6f}")

    assert torch.allclose(std[:, 0], std[:, 2]), "Actions 0 and 2 should have same std (group 0)"
    assert torch.allclose(std[:, 1], std[:, 3]), "Actions 1 and 3 should have same std (group 1)"
    assert not torch.allclose(std[:, 0], std[:, 1]), "Actions 0 (group 0) and 1 (group 1) should have different std"
    assert torch.allclose(std[0, 0], torch.tensor(0.5)), f"Action 0 should have std 0.5, got {std[0, 0].item()}"
    assert torch.allclose(std[0, 1], torch.tensor(1.5)), f"Action 1 should have std 1.5, got {std[0, 1].item()}"

    print("✅ PASSED: std sharing verified")
    return True


if __name__ == "__main__":
    test1 = test_group_dependent_std_scalar()
    test2 = test_group_dependent_std_per_action()
    test3 = test_action_std_groups()
    test4 = test_std_sharing_verification()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    if test1 and test2 and test3 and test4:
        print("✅ ALL TESTS PASSED")
        print("\nUsage examples:")
        print("  # All actions share one std")
        print("  policy = {'class_name': 'ActorCriticCNN', 'group_dependent_std': False}")
        print("")
        print("  # Each action has its own std (default)")
        print("  policy = {'class_name': 'ActorCriticCNN', 'group_dependent_std': True}")
        print("")
        print("  # Grouped std: thrust+omega_y share std, omega_x+omega_z share std")
        print("  policy = {")
        print("    'class_name': 'ActorCriticCNN',")
        print("    'group_dependent_std': True,")
        print("    'action_std_groups': [[0, 2], [1, 3]],")
        print("  }")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
