"""Channel-specific CNN and siamese training objective (Maiorana 2021).

Table 1 of the paper defines a shallow 1D CNN that maps a single-channel
frame of 320 samples to a 256-dimensional embedding.  A *separate* network
with this architecture is trained for every electrode ("channel-specific"
modeling, Section 3.1 / 4.1).

The two sub-networks of the siamese architecture share weights and are
optimised with the contrastive loss of Eq. (1):

    L(x1, x2, y) = (1 - y) * 1/2 * D(x1, x2)^2
                 +      y  * 1/2 * max(0, d - D(x1, x2))^2

with y = 0 for a genuine pair (same subject) and y = 1 for an impostor
pair (different subjects), and d the margin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelCNN(nn.Module):
    """Table 1 network: 1 x 320 -> 256-dimensional representation.

    Every convolutional layer is followed by batch normalisation, as stated
    in the caption of Table 1.  ``padding=2`` for the 5-tap convolutions and
    valid convolutions for the 3-tap ones reproduce the exact input/output
    dimensions listed in the table.
    """

    def __init__(self, dropout: float = 0.5, embedding_dim: int = 256):
        super().__init__()
        # L1: Conv (1x5x1) x16, pad 2   -> 1 x 320 x 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        # L3: max-pool 1x3 -> 106
        # L4: Conv (1x5x16) x32, pad 2 -> 1 x 106 x 32
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        # L6: max-pool 1x3 -> 35
        # L7: Conv (1x3x32) x64 -> 1 x 33 x 64
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        # L9: max-pool 1x3 -> 11
        # L10: Conv (1x3x64) x128 -> 1 x 9 x 128
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(128)
        # L12: max-pool 1x3 -> 3
        self.pool = nn.MaxPool1d(kernel_size=3)
        # L13: dropout
        self.dropout = nn.Dropout(dropout)
        # L14: Conv (1x3x128) x256 -> 1 x 1 x 256
        self.conv5 = nn.Conv1d(128, embedding_dim, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 320) -> (B, 256)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn5(self.conv5(x))        # L14 (no activation, dims 1 x 1 x 256)
        return x.flatten(1)


def contrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor,
                     labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss of Eq. (1).

    ``labels`` = 0 for genuine pairs (same subject), 1 for impostor pairs.
    """
    d = torch.pairwise_distance(emb1, emb2, p=2)
    positive = (1.0 - labels) * 0.5 * d.pow(2)
    negative = labels * 0.5 * F.relu(margin - d).pow(2)
    return (positive + negative).mean()
