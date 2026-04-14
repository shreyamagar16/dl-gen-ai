"""Model definitions for audio genre classification.

Contains :class:`CNNModel`, a convolutional network that takes single-channel
mel spectrograms (shape ``N × 1 × H × W``) and outputs logits for 10 genres.
"""

import torch.nn as nn


class CNNModel(nn.Module):
    """Two-stage CNN: conv blocks for local patterns, then fully-connected layers."""

    def __init__(self):
        super().__init__()

        # Stack of conv + ReLU + pool; doubles channels and halves spatial size each block
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Map flattened features to hidden dim, then to num_classes (10)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # x: batch of mel spectrograms, shape (N, 1, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x
