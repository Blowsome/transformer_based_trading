import torch
from torch import nn

import config


class StockTransformer(nn.Module):
    """60 days x N features -> predicted next-day return (in standardized units)."""

    def __init__(self, n_features: int):
        super().__init__()
        # the "embedding" for continuous inputs: a trainable linear map, features stay fixed
        self.input_proj = nn.Linear(n_features, config.D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.WINDOW, config.D_MODEL))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.N_HEADS, dim_feedforward=config.D_FF,
            dropout=config.DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.N_LAYERS,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(config.D_MODEL), nn.Linear(config.D_MODEL, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_emb
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)  # read out the most recent day
