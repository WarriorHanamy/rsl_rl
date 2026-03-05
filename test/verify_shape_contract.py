"""Standalone test to verify shape contract without pytest dependency."""

import sys

sys.path.insert(0, "/home/rec/server/rsl_rl")

import torch
from tensordict import TensorDict
from rsl_rl.storage.rollout_storage import RolloutStorage


def test_feedforward_shape():
    """Test Feedforward: shape should be (T*N, dim)."""
    print("\n" + "=" * 80)
    print("TEST 1: Feedforward Mini-Batch Generator Shape")
    print("=" * 80)

    num_envs = 4
    num_transitions = 8
    obs_dim = 10
    action_dim = 3
    device = "cpu"

    obs_example = TensorDict(
        {"policy": torch.zeros(num_envs, obs_dim)},
        batch_size=[num_envs],
        device=device,
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs=obs_example,
        actions_shape=(action_dim,),
        device=device,
    )

    for step in range(num_transitions):
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict(
            {"policy": torch.randn(num_envs, obs_dim)},
            batch_size=[num_envs],
            device=device,
        )
        transition.actions = torch.randn(num_envs, action_dim)
        transition.rewards = torch.randn(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.uint8)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.action_mean = torch.randn(num_envs, action_dim)
        transition.action_sigma = torch.abs(torch.randn(num_envs, action_dim))
        storage.add_transition(transition)

    num_mini_batches = 2
    generator = storage.mini_batch_generator(num_mini_batches, num_epochs=1)
    batch = next(generator)

    obs_batch = batch[0]
    actions_batch = batch[1]
    hidden_states = batch[8]
    masks = batch[9]

    # TensorDict needs to access the actual tensor
    if isinstance(obs_batch, TensorDict):
        obs_tensor = obs_batch["policy"]
    else:
        obs_tensor = obs_batch

    expected_batch_size = (num_transitions * num_envs) // num_mini_batches

    print(f"Storage shape: T={num_transitions}, N={num_envs}")
    print(f"Expected batch size: {expected_batch_size} (= {num_transitions} * {num_envs} // {num_mini_batches})")
    print(f"\nActual shapes:")
    print(f"  obs_batch (TensorDict): {obs_batch.shape if isinstance(obs_batch, TensorDict) else 'N/A'}")
    print(f"  obs_tensor:              {obs_tensor.shape} (expected: ({expected_batch_size}, {obs_dim}))")
    print(f"  actions_batch:           {actions_batch.shape} (expected: ({expected_batch_size}, {action_dim}))")
    print(f"  hidden_states:           {hidden_states}")
    print(f"  masks:                   {masks}")

    success = True
    if obs_tensor.shape != (expected_batch_size, obs_dim):
        print(f"\n❌ FAILED: obs_tensor shape mismatch!")
        success = False
    if actions_batch.shape != (expected_batch_size, action_dim):
        print(f"\n❌ FAILED: actions_batch shape mismatch!")
        success = False
    if hidden_states != (None, None):
        print(f"\n❌ FAILED: hidden_states should be (None, None)!")
        success = False
    if masks is not None:
        print(f"\n❌ FAILED: masks should be None!")
        success = False

    if success:
        print(f"\n✅ PASSED: Feedforward generator correctly flattens to (T*N, dim)")

    return success


def test_recurrent_shape():
    """Test Recurrent: shape should be (T, batch, dim)."""
    print("\n" + "=" * 80)
    print("TEST 2: Recurrent Mini-Batch Generator Shape")
    print("=" * 80)

    num_envs = 4
    num_transitions = 8
    obs_dim = 10
    action_dim = 3
    device = "cpu"

    obs_example = TensorDict(
        {"policy": torch.zeros(num_envs, obs_dim)},
        batch_size=[num_envs],
        device=device,
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs=obs_example,
        actions_shape=(action_dim,),
        device=device,
    )

    for step in range(num_transitions):
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict(
            {"policy": torch.randn(num_envs, obs_dim)},
            batch_size=[num_envs],
            device=device,
        )
        transition.actions = torch.randn(num_envs, action_dim)
        transition.rewards = torch.randn(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.uint8)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.action_mean = torch.randn(num_envs, action_dim)
        transition.action_sigma = torch.abs(torch.randn(num_envs, action_dim))

        if step == 3:
            transition.dones[0] = 1
        if step == 5:
            transition.dones[1] = 1
        if step == 7:
            transition.dones[:] = 1

        transition.hidden_states = (
            torch.randn(1, num_envs, 256),
            torch.randn(1, num_envs, 256),
        )

        storage.add_transition(transition)

    num_mini_batches = 2
    generator = storage.recurrent_mini_batch_generator(num_mini_batches, num_epochs=1)
    batch = next(generator)

    obs_batch = batch[0]
    actions_batch = batch[1]
    hidden_states = batch[8]
    masks = batch[9]

    # TensorDict needs to access the actual tensor
    if isinstance(obs_batch, TensorDict):
        obs_tensor = obs_batch["policy"]
    else:
        obs_tensor = obs_batch

    print(f"Storage shape: T={num_transitions}, N={num_envs}")
    print(f"\nActual shapes:")
    print(f"  obs_batch (TensorDict): {obs_batch.shape if isinstance(obs_batch, TensorDict) else 'N/A'}")
    print(f"  obs_tensor:              {obs_tensor.shape}")
    print(f"  actions_batch:           {actions_batch.shape}")
    print(f"  hidden_states[0]:        {hidden_states[0].shape if hidden_states[0] is not None else None}")
    print(f"  hidden_states[1]:        {hidden_states[1].shape if hidden_states[1] is not None else None}")
    print(f"  masks:                   {masks.shape}")

    success = True
    if len(obs_tensor.shape) != 3:
        print(f"\n❌ FAILED: obs_tensor should be 3D!")
        success = False
    else:
        print(f"\n  Time dimension:          {obs_tensor.shape[0]} (max: {num_transitions})")
        print(f"  Trajectories:            {obs_tensor.shape[1]}")
        print(f"  Obs dimension:           {obs_tensor.shape[2]}")

    if obs_tensor.shape[0] > num_transitions:
        print(f"\n❌ FAILED: Time dimension exceeds max!")
        success = False
    if obs_tensor.shape[2] != obs_dim:
        print(f"\n❌ FAILED: Obs dimension mismatch!")
        success = False
    if hidden_states == (None, None):
        print(f"\n❌ FAILED: hidden_states should not be (None, None)!")
        success = False
    if masks is None:
        print(f"\n❌ FAILED: masks should not be None!")
        success = False
    elif len(masks.shape) != 2:
        print(f"\n❌ FAILED: masks should be 2D!")
        success = False

    if success:
        print(f"\n✅ PASSED: Recurrent generator correctly preserves time dimension (T, batch, dim)")

    return success


def test_comparison():
    """Compare feedforward vs recurrent shapes."""
    print("\n" + "=" * 80)
    print("TEST 3: Shape Contract Comparison")
    print("=" * 80)

    num_envs = 4
    num_transitions = 8
    obs_dim = 10
    action_dim = 3
    device = "cpu"

    obs_example = TensorDict(
        {"policy": torch.zeros(num_envs, obs_dim)},
        batch_size=[num_envs],
        device=device,
    )

    storage_ff = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs=obs_example,
        actions_shape=(action_dim,),
        device=device,
    )

    for step in range(num_transitions):
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict(
            {"policy": torch.randn(num_envs, obs_dim)},
            batch_size=[num_envs],
            device=device,
        )
        transition.actions = torch.randn(num_envs, action_dim)
        transition.rewards = torch.randn(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.uint8)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.action_mean = torch.randn(num_envs, action_dim)
        transition.action_sigma = torch.abs(torch.randn(num_envs, action_dim))
        storage_ff.add_transition(transition)

    storage_rec = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs=obs_example,
        actions_shape=(action_dim,),
        device=device,
    )

    for step in range(num_transitions):
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict(
            {"policy": torch.randn(num_envs, obs_dim)},
            batch_size=[num_envs],
            device=device,
        )
        transition.actions = torch.randn(num_envs, action_dim)
        transition.rewards = torch.randn(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.uint8)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.action_mean = torch.randn(num_envs, action_dim)
        transition.action_sigma = torch.abs(torch.randn(num_envs, action_dim))

        if step == 3:
            transition.dones[0] = 1
        if step == 5:
            transition.dones[1] = 1
        if step == 7:
            transition.dones[:] = 1

        transition.hidden_states = (
            torch.randn(1, num_envs, 256),
            torch.randn(1, num_envs, 256),
        )

        storage_rec.add_transition(transition)

    ff_gen = storage_ff.mini_batch_generator(num_mini_batches=2, num_epochs=1)
    ff_batch = next(ff_gen)
    ff_obs = ff_batch[0]

    rec_gen = storage_rec.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1)
    rec_batch = next(rec_gen)
    rec_obs = rec_batch[0]

    # Access actual tensors from TensorDict
    if isinstance(ff_obs, TensorDict):
        ff_obs_tensor = ff_obs["policy"]
    else:
        ff_obs_tensor = ff_obs

    if isinstance(rec_obs, TensorDict):
        rec_obs_tensor = rec_obs["policy"]
    else:
        rec_obs_tensor = rec_obs

    print(f"\nFeedforward obs shape: {ff_obs_tensor.shape}")
    print(f"  Dimensions: {len(ff_obs_tensor.shape)}D (T*N flattened)")
    print(f"  Total samples: {ff_obs_tensor.shape[0]}")

    print(f"\nRecurrent obs shape: {rec_obs_tensor.shape}")
    print(f"  Dimensions: {len(rec_obs_tensor.shape)}D (T preserved)")
    print(f"  Time steps: {rec_obs_tensor.shape[0]}")
    print(f"  Trajectories: {rec_obs_tensor.shape[1]}")
    print(f"  Total samples: {rec_obs_tensor.shape[0] * rec_obs_tensor.shape[1]}")

    success = True
    if len(ff_obs_tensor.shape) != 2:
        print(f"\n❌ FAILED: Feedforward should be 2D!")
        success = False
    if len(rec_obs_tensor.shape) != 3:
        print(f"\n❌ FAILED: Recurrent should be 3D!")
        success = False

    if success:
        print(f"\n✅ PASSED: Shape contracts are different as expected")
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: Shape Contract Verified")
        print(f"{'=' * 80}")
        print(f"Feedforward: (T*N, dim) - Time and environment dimensions flattened")
        print(f"Recurrent:   (T, batch, dim) - Time dimension preserved")
        print(f"{'=' * 80}")

    return success


if __name__ == "__main__":
    test1 = test_feedforward_shape()
    test2 = test_recurrent_shape()
    test3 = test_comparison()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED")
        print("\nShape contract verified:")
        print("  - Feedforward: (T*N, dim)")
        print("  - Recurrent:   (T, batch, dim)")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
