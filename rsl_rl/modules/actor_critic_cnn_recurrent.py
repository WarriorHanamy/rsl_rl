# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, HiddenState, Memory


class ActorCriticCNNRecurrent(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        group_dependent_std: bool = True,
        action_std_groups: list[list[int]] | None = None,
        dim_hidden_input: int = 128,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticCNNRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.dim_hidden_input = dim_hidden_input

        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_2d = []
        actor_in_channels_2d = []
        self.actor_obs_groups_2d = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 4:
                self.actor_obs_groups_2d.append(obs_group)
                actor_in_dims_2d.append(obs[obs_group].shape[2:4])
                actor_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        critic_in_dims_2d = []
        critic_in_channels_2d = []
        self.critic_obs_groups_2d = []
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) == 4:
                self.critic_obs_groups_2d.append(obs_group)
                critic_in_dims_2d.append(obs[obs_group].shape[2:4])
                critic_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if self.actor_obs_groups_2d:
            assert actor_cnn_cfg is not None, "An actor CNN configuration is required for 2D actor observations."
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {group: actor_cnn_cfg for group in self.actor_obs_groups_2d}
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D actor observations."
            )

            self.actor_cnns = nn.ModuleDict()
            for idx, obs_group in enumerate(self.actor_obs_groups_2d):
                self.actor_cnns[obs_group] = CNN(
                    input_dim=actor_in_dims_2d[idx],
                    input_channels=actor_in_channels_2d[idx],
                    **actor_cnn_cfg[obs_group],
                )
                print(f"Actor CNN for {obs_group}: {self.actor_cnns[obs_group]}")
                if self.actor_cnns[obs_group].output_channels is not None:
                    raise ValueError("The output of the actor CNN must be flattened before passing it to the MLP.")

            self.actor_cnn_projection = nn.ModuleDict()
            for obs_group in self.actor_obs_groups_2d:
                cnn_out_dim = int(self.actor_cnns[obs_group].output_dim)
                self.actor_cnn_projection[obs_group] = nn.Linear(cnn_out_dim, self.dim_hidden_input)
                print(f"Actor CNN Projection for {obs_group}: {cnn_out_dim} -> {self.dim_hidden_input}")
        else:
            self.actor_cnns = None

        if num_actor_obs_1d > 0:
            self.actor_state_encoder = nn.Linear(num_actor_obs_1d, self.dim_hidden_input)
            print(f"Actor State Encoder: Linear({num_actor_obs_1d} -> {self.dim_hidden_input})")
        else:
            self.actor_state_encoder = None

        self.memory_a = Memory(self.dim_hidden_input, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        if self.critic_obs_groups_2d:
            assert critic_cnn_cfg is not None, "A critic CNN configuration is required for 2D critic observations."
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {group: critic_cnn_cfg for group in self.critic_obs_groups_2d}
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D critic observations."
            )

            self.critic_cnns = nn.ModuleDict()
            for idx, obs_group in enumerate(self.critic_obs_groups_2d):
                self.critic_cnns[obs_group] = CNN(
                    input_dim=critic_in_dims_2d[idx],
                    input_channels=critic_in_channels_2d[idx],
                    **critic_cnn_cfg[obs_group],
                )
                print(f"Critic CNN for {obs_group}: {self.critic_cnns[obs_group]}")
                if self.critic_cnns[obs_group].output_channels is not None:
                    raise ValueError("The output of the critic CNN must be flattened before passing it to the MLP.")

            self.critic_cnn_projection = nn.ModuleDict()
            for obs_group in self.critic_obs_groups_2d:
                cnn_out_dim = int(self.critic_cnns[obs_group].output_dim)
                self.critic_cnn_projection[obs_group] = nn.Linear(cnn_out_dim, self.dim_hidden_input)
                print(f"Critic CNN Projection for {obs_group}: {cnn_out_dim} -> {self.dim_hidden_input}")
        else:
            self.critic_cnns = None

        if num_critic_obs_1d > 0:
            self.critic_state_encoder = nn.Linear(num_critic_obs_1d, self.dim_hidden_input)
            print(f"Critic State Encoder: Linear({num_critic_obs_1d} -> {self.dim_hidden_input})")
        else:
            self.critic_state_encoder = None

        self.memory_c = Memory(self.dim_hidden_input, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        self.group_dependent_std = group_dependent_std
        self.action_std_groups = action_std_groups
        self.num_actions = num_actions

        if not group_dependent_std:
            num_std = 1
            self.register_buffer("std_group_map", None)
        elif action_std_groups is None:
            num_std = num_actions
            self.register_buffer("std_group_map", None)
        else:
            num_std = len(action_std_groups)
            std_group_map = torch.zeros(num_actions, dtype=torch.long)
            for group_idx, group in enumerate(action_std_groups):
                for action_idx in group:
                    std_group_map[action_idx] = group_idx
            self.register_buffer("std_group_map", std_group_map)

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_std))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_std)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None

        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _encode_cnn_features(
        self,
        cnn_module: nn.Module,
        proj_module: nn.Module,
        obs_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if obs_tensor.ndim == 5:
            time_steps, batch_size, channels, height, width = obs_tensor.shape
            obs_flat = obs_tensor.reshape(time_steps * batch_size, channels, height, width)
            cnn_feat = cnn_module(obs_flat)
            cnn_feat = proj_module(cnn_feat)
            return cnn_feat.reshape(time_steps, batch_size, -1)
        if obs_tensor.ndim == 4:
            cnn_feat = cnn_module(obs_tensor)
            return proj_module(cnn_feat)
        raise ValueError(f"Expected image tensor with 4 or 5 dims, got: {obs_tensor.shape}")

    def _get_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            std = self.std
        else:
            std = torch.exp(self.log_std)

        if not self.group_dependent_std:
            std = std.expand_as(mean)
        elif self.std_group_map is not None:
            std = std[self.std_group_map].expand_as(mean)
        else:
            std = std.expand_as(mean)

        return std

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        std = self._get_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        cnn_enc_list = []
        if self.actor_cnns is not None:
            for obs_group in self.actor_obs_groups_2d:
                cnn_feat = self._encode_cnn_features(
                    self.actor_cnns[obs_group],
                    self.actor_cnn_projection[obs_group],
                    cnn_obs[obs_group],
                )
                cnn_enc_list.append(cnn_feat)
        if len(cnn_enc_list) > 1:
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
        elif len(cnn_enc_list) == 1:
            cnn_enc = cnn_enc_list[0]
        else:
            cnn_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        if self.actor_state_encoder is not None and mlp_obs.shape[-1] > 0:
            state_enc = self.actor_state_encoder(mlp_obs)
        else:
            state_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        fused_features = cnn_enc + state_enc
        out_mem = self.memory_a(fused_features, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        cnn_enc_list = []
        if self.actor_cnns is not None:
            for obs_group in self.actor_obs_groups_2d:
                cnn_feat = self._encode_cnn_features(
                    self.actor_cnns[obs_group],
                    self.actor_cnn_projection[obs_group],
                    cnn_obs[obs_group],
                )
                cnn_enc_list.append(cnn_feat)
        if len(cnn_enc_list) > 1:
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
        elif len(cnn_enc_list) == 1:
            cnn_enc = cnn_enc_list[0]
        else:
            cnn_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        if self.actor_state_encoder is not None and mlp_obs.shape[-1] > 0:
            state_enc = self.actor_state_encoder(mlp_obs)
        else:
            state_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        fused_features = cnn_enc + state_enc
        out_mem = self.memory_a(fused_features).squeeze(0)
        return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        cnn_enc_list = []
        if self.critic_cnns is not None:
            for obs_group in self.critic_obs_groups_2d:
                cnn_feat = self._encode_cnn_features(
                    self.critic_cnns[obs_group],
                    self.critic_cnn_projection[obs_group],
                    cnn_obs[obs_group],
                )
                cnn_enc_list.append(cnn_feat)
        if len(cnn_enc_list) > 1:
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
        elif len(cnn_enc_list) == 1:
            cnn_enc = cnn_enc_list[0]
        else:
            cnn_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        if self.critic_state_encoder is not None and mlp_obs.shape[-1] > 0:
            state_enc = self.critic_state_encoder(mlp_obs)
        else:
            state_enc = torch.zeros((*mlp_obs.shape[:-1], self.dim_hidden_input), device=mlp_obs.device)

        fused_features = cnn_enc + state_enc
        out_mem = self.memory_c(fused_features, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.actor_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.critic_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
