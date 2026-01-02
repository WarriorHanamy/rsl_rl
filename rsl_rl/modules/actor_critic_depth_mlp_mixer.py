# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import MLP, EmpiricalNormalization, DepthMLPMixer

from .actor_critic import ActorCritic


class ActorCriticDepthMLPMixer(ActorCritic):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        # 这里接收 mixer 的配置，例如 pool_kernel, patch_size, hidden_dim 等
        actor_mixer_cfg: dict[str, dict] | dict | None = None,
        critic_mixer_cfg: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDepthMLPMixer.__init__ got unexpected arguments: "
                + str([key for key in kwargs])
            )
        
        # 调用父类的父类构造函数 (ActorCritic 的基类通常是 nn.Module)
        super(ActorCritic, self).__init__()

        self.obs_groups = obs_groups
        
        # --- 解析 Actor 观测维度 ---
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_2d = []
        actor_in_channels_2d = []
        self.actor_obs_groups_2d = []
        
        for obs_group in obs_groups["policy"]:
            shape = obs[obs_group].shape
            if len(shape) == 4:  # [B, C, H, W]
                self.actor_obs_groups_2d.append(obs_group)
                actor_in_dims_2d.append(shape[2:4]) # (H, W)
                actor_in_channels_2d.append(shape[1])
            elif len(shape) == 2:  # [B, D]
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += shape[-1]

        # --- 解析 Critic 观测维度 ---
        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        critic_in_dims_2d = []
        critic_in_channels_2d = []
        self.critic_obs_groups_2d = []
        
        for obs_group in obs_groups["critic"]:
            shape = obs[obs_group].shape
            if len(shape) == 4:
                self.critic_obs_groups_2d.append(obs_group)
                critic_in_dims_2d.append(shape[2:4])
                critic_in_channels_2d.append(shape[1])
            elif len(shape) == 2:
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += shape[-1]

        # --- 实例化 Actor DepthMLPMixers ---
        self.actor_mixers = nn.ModuleDict()
        actor_encoding_dim = 0
        if self.actor_obs_groups_2d:
            assert actor_mixer_cfg is not None, "Actor mixer configuration is required."
            if not all(isinstance(v, dict) for v in actor_mixer_cfg.values()):
                actor_mixer_cfg = {group: actor_mixer_cfg for group in self.actor_obs_groups_2d}

            for idx, obs_group in enumerate(self.actor_obs_groups_2d):
                self.actor_mixers[obs_group] = DepthMLPMixer(
                    in_channels=actor_in_channels_2d[idx],
                    image_size=actor_in_dims_2d[idx],
                    **actor_mixer_cfg[obs_group]
                )

                assert actor_mixer_cfg is not None and actor_mixer_cfg[obs_group]['flatten'], "Actor mixer must have flatten=True to compute encoding dim."
                actor_encoding_dim += self.actor_mixers[obs_group].output_dim

        # --- 实例化 Critic DepthMLPMixers ---
        self.critic_mixers = nn.ModuleDict()
        critic_encoding_dim = 0
        if self.critic_obs_groups_2d:
            assert critic_mixer_cfg is not None, "Critic mixer configuration is required."
            if not all(isinstance(v, dict) for v in critic_mixer_cfg.values()):
                critic_mixer_cfg = {group: critic_mixer_cfg for group in self.critic_obs_groups_2d}

            for idx, obs_group in enumerate(self.critic_obs_groups_2d):
                self.critic_mixers[obs_group] = DepthMLPMixer(
                    in_channels=critic_in_channels_2d[idx],
                    image_size=critic_in_dims_2d[idx],
                    **critic_mixer_cfg[obs_group]
                )
                m_cfg = critic_mixer_cfg[obs_group]
                h_eff = critic_in_dims_2d[idx][0] // m_cfg.get('pool_kernel', 2) // m_cfg.get('patch_size', 4)
                w_eff = critic_in_dims_2d[idx][1] // m_cfg.get('pool_kernel', 2) // m_cfg.get('patch_size', 4)
                num_tokens = h_eff * w_eff
                critic_encoding_dim += num_tokens * m_cfg.get('hidden_dim', 128)

        # --- Actor/Critic MLP Heads ---
        self.state_dependent_std = state_dependent_std
        actor_input_dim = num_actor_obs_1d + actor_encoding_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        
        self.critic = MLP(num_critic_obs_1d + critic_encoding_dim, 1, critic_hidden_dims, activation)

        # --- Normalization & Noise ---
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d) if critic_obs_normalization else nn.Identity()

        self.noise_std_type = noise_std_type
        if not self.state_dependent_std:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            else:
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_mixers:
            mixer_enc_list = []
            for obs_group in self.actor_obs_groups_2d:
                # Mixer 返回 [B, S, C], 需要展平为 [B, S*C]
                z = self.actor_mixers[obs_group](cnn_obs[obs_group])
                mixer_enc_list.append(z.flatten(1)) 
            
            mixer_enc = torch.cat(mixer_enc_list, dim=-1)
            mlp_obs = torch.cat([mlp_obs, mixer_enc], dim=-1)

        super()._update_distribution(mlp_obs)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample() # type: ignore

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_mixers:
            mixer_enc_list = [self.actor_mixers[g](cnn_obs[g]).flatten(1) for g in self.actor_obs_groups_2d]
            mlp_obs = torch.cat([mlp_obs, torch.cat(mixer_enc_list, dim=-1)], dim=-1)

        return self.actor(mlp_obs)[..., 0, :] if self.state_dependent_std else self.actor(mlp_obs)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_mixers:
            mixer_enc_list = [self.critic_mixers[g](cnn_obs[g]).flatten(1) for g in self.critic_obs_groups_2d]
            mlp_obs = torch.cat([mlp_obs, torch.cat(mixer_enc_list, dim=-1)], dim=-1)

        return self.critic(mlp_obs)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_1d = torch.cat([obs[g] for g in self.actor_obs_groups_1d], dim=-1)
        obs_2d = {g: obs[g] for g in self.actor_obs_groups_2d}
        return obs_1d, obs_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_1d = torch.cat([obs[g] for g in self.critic_obs_groups_1d], dim=-1)
        obs_2d = {g: obs[g] for g in self.critic_obs_groups_2d}
        return obs_1d, obs_2d