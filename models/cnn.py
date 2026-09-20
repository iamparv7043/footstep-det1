"""
models/cnn.py

Heavier CNN for footstep detection than the earlier lightweight
prototype. Four convolutional blocks (32-64-128-256 channels) instead
of three, with Dropout2d between blocks for regularization, Global
Average Pooling, and an optional auxiliary scalar input (low-frequency
energy ratio) concatenated before the classifier head -- this gives the
model an explicit, hand-engineered signal for the footstep-vs-broadband
distinction on top of what it learns from the spectrogram alone.

Input:  log-mel spectrogram (batch, 1, n_mels, n_frames), e.g. (B,1,64,51)
        + optional aux (batch, 1) -- low_freq_ratio
Output: (batch, 2) raw logits -- [no_footstep, footstep]
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(dropout * 0.5),  # lighter dropout inside conv stack
    )


class FootstepCNN(nn.Module):
    def __init__(self, n_mels: int = 64, dropout: float = 0.4,
                 use_aux_features: bool = True,
                 channels=(32, 64, 128, 256)):
        super().__init__()
        self.use_aux_features = use_aux_features
        self.dropout_rate = dropout

        c1, c2, c3, c4 = channels
        self.conv_block = nn.Sequential(
            conv_block(1, c1, dropout),
            conv_block(c1, c2, dropout),
            conv_block(c2, c3, dropout),
            conv_block(c3, c4, dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        head_in = c4 + (1 if use_aux_features else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.gap(x).flatten(1)
        if self.use_aux_features:
            if aux is None:
                aux = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, aux], dim=1)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        """Parameter count grouped by layer type, for the training report."""
        breakdown = {}
        for module in self.modules():
            cls = type(module).__name__
            n = sum(p.numel() for p in module.parameters(recurse=False))
            if n > 0:
                breakdown[cls] = breakdown.get(cls, 0) + n
        return breakdown


if __name__ == "__main__":
    model = FootstepCNN(n_mels=64, dropout=0.4, use_aux_features=True)
    dummy_x = torch.randn(4, 1, 64, 51)
    dummy_aux = torch.rand(4, 1)
    out = model(dummy_x, dummy_aux)
    print("Output shape:", out.shape)
    print("Total params:", f"{model.count_params():,}")
    print("Breakdown:", model.param_breakdown())
