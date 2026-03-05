from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rsl_rl.networks.cnn import CNN
from rsl_rl.networks.mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class RecurrentCNN(nn.Module):
    """Recurrent actor with configurable CNN encoder, GRU, and head.

    Configuration is done via ``cnn_cfg`` and ``network_cfg`` dictionaries:

    .. code-block:: python

        network_cfg = {
            "activation": "elu",  # shared activation function
            "gru": {
                "dim_hidden": 192,  # also used for projection layers
                "biased": True,
            },
            "mlp": {
                "hidden_dims": None,  # None means [dim_hidden]
                "activation": None,   # None means inherit from shared activation
            },
        }
    """

    def __init__(
        self,
        input_dim: tuple[int, int] = (12, 16),
        input_channels: int = 1,
        dim_obs: int = 9,
        dim_action: int = 4,
        cnn_cfg: dict[str, Any] | None = None,
        network_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Default CNN configuration
        default_cnn_cfg = {
            "output_channels": [32, 64, 128],
            "kernel_size": [2, 3, 3],
            "stride": [2, 1, 1],
            "padding": "none",
            "norm": "none",
            "activation": "lrelu",
            "bias": False,
        }
        resolved_cnn_cfg = default_cnn_cfg if cnn_cfg is None else {**default_cnn_cfg, **cnn_cfg}
        resolved_cnn_cfg["flatten"] = True

        # Default network configuration
        default_network_cfg = {
            "activation": "elu",
            "gru": {"dim_hidden": 192, "biased": True},
            "mlp": {"hidden_dims": None, "activation": None},
        }
        resolved_cfg = default_network_cfg if network_cfg is None else {**default_network_cfg, **network_cfg}

        # Resolve sub-configurations with merge
        gru_cfg = {**default_network_cfg["gru"], **resolved_cfg.get("gru", {})}
        mlp_cfg = {**default_network_cfg["mlp"], **resolved_cfg.get("mlp", {})}

        dim_hidden = gru_cfg["dim_hidden"]

        # Build CNN encoder
        cnn = CNN(input_dim=input_dim, input_channels=input_channels, **resolved_cnn_cfg)
        dim_cnn_feat = int(cnn.output_dim)

        # Build projection layers (dim_hidden is used for both proj and gru)
        self.stem = nn.Sequential(
            cnn,
            nn.Linear(dim_cnn_feat, dim_hidden, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, dim_hidden)

        # Activation
        activation_function = resolve_nn_activation(resolved_cfg["activation"])
        self.activation = activation_function

        # GRU
        self.dim_gru_hidden = dim_hidden
        self.gru = nn.GRUCell(dim_hidden, dim_hidden, bias=gru_cfg["biased"])

        # Head MLP
        hidden_dims = list(mlp_cfg["hidden_dims"]) if mlp_cfg["hidden_dims"] is not None else [dim_hidden]
        mlp_activation = mlp_cfg["activation"] or resolved_cfg["activation"]
        self.head = MLP(
            input_dim=dim_hidden,
            output_dim=dim_action,
            hidden_dims=hidden_dims,
            activation=mlp_activation,
        )

    def reset(self) -> None:
        pass

    def _forward_gru_cell(self, x: torch.Tensor, hx: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = torch.zeros(x.shape[0], self.dim_gru_hidden, device=x.device, dtype=x.dtype)
        hx = self.gru(x, hx)  # type: ignore[arg-type]
        return hx, hx

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        img_feat = self.stem(x)
        fused = img_feat + self.v_proj(v)
        fused = self.activation(fused)

        out, h_next = self._forward_gru_cell(fused, hx)

        action = self.head(self.activation(out))
        return action, None, h_next
