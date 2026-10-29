import torch
from torch import nn

import config


class StockTransformer(nn.Module):
    """One day of stocks [n_stocks, WINDOW, F] -> one score per stock (higher = expected to beat the others).

    1. a transformer along time, separately for each stock (what has this stock been doing?)
    2. optional: one attention layer across all stocks of the day (how does it compare with the others?)
    """

    def __init__(self, n_features: int):
        super().__init__()
        # the "embedding" for continuous inputs: a trainable linear map, features stay fixed
        self.input_proj = nn.Linear(n_features, config.D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.WINDOW, config.D_MODEL))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = lambda: nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.N_HEADS, dim_feedforward=config.D_FF,
            dropout=config.DROPOUT, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer(), num_layers=config.N_LAYERS, enable_nested_tensor=False)
        self.cross_stock = layer() if config.CROSS_STOCK else None
        self.head = nn.Sequential(nn.LayerNorm(config.D_MODEL), nn.Linear(config.D_MODEL, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.input_proj(x) + self.pos_emb)[:, -1]  # [n, D]: each stock as of the last day
        if self.cross_stock is not None:
            h = self.cross_stock(h.unsqueeze(0)).squeeze(0)        # stocks of the day attend to each other
        return self.head(h).squeeze(-1)
