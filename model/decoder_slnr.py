from __future__ import annotations

import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderSLNR(nn.Module):
    def __init__(self, config, neural_points) -> None:
        super().__init__()
        self.config = config
        self.out_dim = 1
        self._neural_points_ref = weakref.ref(neural_points)

    def _map_module(self):
        neural_points = self._neural_points_ref()
        if neural_points is None:
            return None
        return neural_points.backend.neural_map

    def mlp(self, features: torch.Tensor) -> torch.Tensor:
        map_module = self._map_module()
        if map_module is None:
            raise RuntimeError("SLNR decoder is not initialized because the backend neural map is missing.")

        h = features
        for layer_idx, layer in enumerate(map_module.sdf_net):
            h = layer(h)
            if layer_idx != map_module.num_layers - 1:
                h = F.relu(h, inplace=True)
        return h

    def sdf(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(1)
