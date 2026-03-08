# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState


class StudentTeacher(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        group_dependent_std: bool = True,
        action_std_groups: list[list[int]] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False

        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

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

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

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

    def _update_distribution(self, obs: TensorDict) -> None:
        mean = self.student(obs)
        std = self._get_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters.
        """
        if any("actor" in key for key in state_dict):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any("student" in key for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
