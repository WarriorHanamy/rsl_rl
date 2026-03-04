# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("swanlab package is required to log to SwanLab.") from None


class SwanlabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config.") from None

        try:
            workspace = cfg.get("swanlab_workspace")
        except KeyError:
            workspace = None

        try:
            run_id = cfg.get("swanlab_run_id")
        except KeyError:
            run_id = None

        swanlab.init(project=project, name=run_name, workspace=workspace, run_id=run_id)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        swanlab.config.update({"runner_cfg": train_cfg})
        swanlab.config.update({"policy_cfg": train_cfg["policy"]})
        swanlab.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            swanlab.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            swanlab.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        swanlab.finish()

    def save_model(self, model_path: str, it: int) -> None:
        swanlab.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        swanlab.save(path, base_path=os.path.dirname(path))
