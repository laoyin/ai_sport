from __future__ import annotations

import torch
import torch.nn as nn


class ConvStem(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ConvTinyTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 55,
        seq_len: int = 32,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        num_action_classes: int = 3,
        num_stage_classes: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stem = ConvStem(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.stage_head = nn.Linear(d_model, num_stage_classes)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_action_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.stem(x)       # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]

        if x.size(1) != self.pos_embed.size(1):
            raise ValueError(f"Unexpected sequence length {x.size(1)}")
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        stage_logits = self.stage_head(x)  # [B, T, C_stage]
        pooled = x.mean(dim=1)             # [B, D]
        action_logits = self.action_head(pooled)
        return {
            "stage_logits": stage_logits,
            "action_logits": action_logits,
        }
