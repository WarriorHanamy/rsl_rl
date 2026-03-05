# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class TestMiniBatchGeneratorShapes:
    """Verify shape contract between Feedforward and Recurrent generators."""

    @pytest.fixture
    def storage_params(self):
        """Mock data parameters for testing."""
        return {
            "num_envs": 4,
            "num_transitions": 8,
            "obs_dim": 10,
            "action_dim": 3,
            "device": "cpu",
        }

    @pytest.fixture
    def feedforward_storage(self, storage_params):
        """Create and populate storage for feedforward (non-recurrent) policy."""
        p = storage_params

        obs = TensorDict(
            {"policy": torch.zeros(p["num_transitions"], p["num_envs"], p["obs_dim"])},
            batch_size=[p["num_transitions"], p["num_envs"]],
            device=p["device"],
        )

        storage = RolloutStorage(
            training_type="rl",
            num_envs=p["num_envs"],
            num_transitions_per_env=p["num_transitions"],
            obs=obs,
            actions_shape=(p["action_dim"],),
            device=p["device"],
        )

        for step in range(p["num_transitions"]):
            transition = RolloutStorage.Transition()
            transition.observations = TensorDict(
                {"policy": torch.randn(p["num_envs"], p["obs_dim"])},
                batch_size=[p["num_envs"]],
                device=p["device"],
            )
            transition.actions = torch.randn(p["num_envs"], p["action_dim"])
            transition.rewards = torch.randn(p["num_envs"])
            transition.dones = torch.zeros(p["num_envs"], dtype=torch.uint8)
            transition.values = torch.randn(p["num_envs"], 1)
            transition.actions_log_prob = torch.randn(p["num_envs"], 1)
            transition.action_mean = torch.randn(p["num_envs"], p["action_dim"])
            transition.action_sigma = torch.abs(torch.randn(p["num_envs"], p["action_dim"]))
            storage.add_transition(transition)

        return storage

    @pytest.fixture
    def recurrent_storage(self, storage_params):
        """Create and populate storage for recurrent policy with done signals."""
        p = storage_params

        obs = TensorDict(
            {"policy": torch.zeros(p["num_transitions"], p["num_envs"], p["obs_dim"])},
            batch_size=[p["num_transitions"], p["num_envs"]],
            device=p["device"],
        )

        storage = RolloutStorage(
            training_type="rl",
            num_envs=p["num_envs"],
            num_transitions_per_env=p["num_transitions"],
            obs=obs,
            actions_shape=(p["action_dim"],),
            device=p["device"],
        )

        for step in range(p["num_transitions"]):
            transition = RolloutStorage.Transition()
            transition.observations = TensorDict(
                {"policy": torch.randn(p["num_envs"], p["obs_dim"])},
                batch_size=[p["num_envs"]],
                device=p["device"],
            )
            transition.actions = torch.randn(p["num_envs"], p["action_dim"])
            transition.rewards = torch.randn(p["num_envs"])
            transition.dones = torch.zeros(p["num_envs"], dtype=torch.uint8)
            transition.values = torch.randn(p["num_envs"], 1)
            transition.actions_log_prob = torch.randn(p["num_envs"], 1)
            transition.action_mean = torch.randn(p["num_envs"], p["action_dim"])
            transition.action_sigma = torch.abs(torch.randn(p["num_envs"], p["action_dim"]))

            if step == 3:
                transition.dones[0] = 1
            if step == 5:
                transition.dones[1] = 1
            if step == 7:
                transition.dones[:] = 1

            transition.hidden_states = (
                torch.randn(1, p["num_envs"], 256),
                torch.randn(1, p["num_envs"], 256),
            )

            storage.add_transition(transition)

        return storage

    def test_feedforward_flatten_shape(self, feedforward_storage, storage_params):
        """Test Feedforward: shape should be (T*N, dim)."""
        p = storage_params
        num_mini_batches = 2
        generator = feedforward_storage.mini_batch_generator(num_mini_batches, num_epochs=1)

        batch = next(generator)
        (
            obs_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states,
            masks,
        ) = batch

        expected_batch_size = (p["num_transitions"] * p["num_envs"]) // num_mini_batches

        assert obs_batch.shape == (
            expected_batch_size,
            p["obs_dim"],
        ), f"Expected obs shape {(expected_batch_size, p['obs_dim'])}, got {obs_batch.shape}"
        assert actions_batch.shape == (
            expected_batch_size,
            p["action_dim"],
        ), f"Expected actions shape {(expected_batch_size, p['action_dim'])}, got {actions_batch.shape}"
        assert hidden_states == (
            None,
            None,
        ), f"Expected hidden_states (None, None), got {hidden_states}"
        assert masks is None, f"Expected masks None, got {masks}"

    def test_recurrent_preserves_time_dim(self, recurrent_storage, storage_params):
        """Test Recurrent: shape should be (T, batch, dim)."""
        p = storage_params
        num_mini_batches = 2
        generator = recurrent_storage.recurrent_mini_batch_generator(num_mini_batches, num_epochs=1)

        batch = next(generator)
        (
            obs_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states,
            masks,
        ) = batch

        assert len(obs_batch.shape) == 3, f"Expected 3D obs tensor, got {len(obs_batch.shape)}D: {obs_batch.shape}"
        assert obs_batch.shape[0] <= p["num_transitions"], (
            f"Time dimension {obs_batch.shape[0]} exceeds max {p['num_transitions']}"
        )
        assert obs_batch.shape[2] == p["obs_dim"], f"Expected obs_dim {p['obs_dim']}, got {obs_batch.shape[2]}"

        assert hidden_states != (None, None), "Expected non-None hidden states"
        assert masks is not None, "Expected non-None masks"
        assert len(masks.shape) == 2, f"Expected 2D masks, got {len(masks.shape)}D: {masks.shape}"

    def test_shape_contract_difference(self, feedforward_storage, recurrent_storage, storage_params):
        """Compare: verify shape difference between feedforward and recurrent."""
        p = storage_params

        ff_gen = feedforward_storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
        ff_batch = next(ff_gen)
        ff_obs = ff_batch[0]

        rec_gen = recurrent_storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1)
        rec_batch = next(rec_gen)
        rec_obs = rec_batch[0]

        assert len(ff_obs.shape) == 2, f"Feedforward obs should be 2D, got {len(ff_obs.shape)}D"
        assert len(rec_obs.shape) == 3, f"Recurrent obs should be 3D, got {len(rec_obs.shape)}D"

        ff_total_samples = ff_obs.shape[0]
        rec_total_samples = rec_obs.shape[0] * rec_obs.shape[1]
        expected_total = p["num_transitions"] * p["num_envs"]

        assert ff_total_samples == expected_total // 2, f"Feedforward batch size mismatch"
        assert rec_total_samples <= expected_total // 2, f"Recurrent batch size mismatch"

    def test_feedforward_random_sampling(self, feedforward_storage):
        """Verify Feedforward data is randomly shuffled."""
        generator = feedforward_storage.mini_batch_generator(num_mini_batches=1, num_epochs=2)

        batch1 = next(generator)
        batch2 = next(generator)

        obs1 = batch1[0]
        obs2 = batch2[0]

        assert not torch.allclose(obs1, obs2), "Expected different samples across epochs (random shuffling)"

    def test_recurrent_sequential_order(self, recurrent_storage):
        """Verify Recurrent preserves temporal order."""
        generator = recurrent_storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1)

        batch = next(generator)
        obs_batch = batch[0]
        masks = batch[9]

        assert obs_batch.shape[0] > 1, "Expected sequence length > 1"

        for t in range(1, obs_batch.shape[0]):
            mask_t = masks[t]
            if mask_t.any():
                num_valid = mask_t.sum().item()
                assert num_valid > 0, f"Invalid trajectory at time step {t}"
