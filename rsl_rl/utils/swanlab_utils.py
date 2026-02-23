# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SwanLab integration for RSL-RL.

SwanLab is an open-source AI training tracking and visualization platform.
Documentation: https://docs.swanlab.cn/en/
"""

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "swanlab package is required to log to SwanLab. Install with: pip install swanlab"
    ) from None


class SwanLabSummaryWriter(SummaryWriter):
    """Summary writer for SwanLab.

    This class extends TensorBoard's SummaryWriter to also log metrics to SwanLab.
    It follows the same pattern as WandbSummaryWriter and NeptuneSummaryWriter.

    Configuration:
      - swanlab_project: Project name in SwanLab (required)
      - swanlab_workspace: Workspace name (optional, uses default if not set)
      - experiment_name: Experiment name (optional, uses log_dir name if not set)
      - SWANLAB_API_KEY: Environment variable for API key (optional for public projects)
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name from log_dir
        run_name = os.path.split(log_dir)[-1]

        # Get SwanLab project (required)
        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError(
                "Please specify swanlab_project in the runner config, e.g. swanlab_project: 'drone-racer'"
            ) from None

        # Get workspace (optional)
        workspace = cfg.get("swanlab_workspace", None)

        # Get experiment name (use run_name if not specified)
        experiment_name = cfg.get("experiment_name", run_name)

        # Get description if provided
        description = cfg.get("swanlab_description", None)

        # Initialize SwanLab
        swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=experiment_name,
            description=description,
            config={"log_dir": log_dir},
        )

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Store training and environment configuration to SwanLab."""
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
        """Log a scalar to both TensorBoard and SwanLab."""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """Finish the SwanLab run."""
        swanlab.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """Save model artifact to SwanLab."""
        swanlab.save(model_path)

    def save_file(self, path: str) -> None:
        """Save a file to SwanLab."""
        swanlab.save(path)
