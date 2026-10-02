"""CNN, optimizer and loss supported by paper section II-B2."""
import torch
from torch import nn


class CoSP(nn.Module):
    """Input [batch,1,120,109]; output logits [batch,1].

    Conv-BN-ReLU-pool and same padding follow the original code. Four pools
    produce 7x6; the paper's intermediate 15x15 is inconsistent with halving.
    """
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels, kernel in zip((32, 64, 128, 256), (7, 5, 3, 3)):
            layers.extend([nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2),
                           nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 7 * 6, 128),
                                  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(128, 1))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 120, 109):
            raise ValueError('Expected input shape [batch,1,120,109]')
        return self.head(self.features(x))


def training_components(model):
    """Adam 1e-5, pure binary cross-entropy; no PP proxy or AdamW decay."""
    return torch.optim.Adam(model.parameters(), lr=1e-5), nn.BCEWithLogitsLoss()


def train_step(model, optimizer, criterion, inputs, labels):
    """One explicit batch step; does not choose data, splits, epochs or seeds."""
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    targets = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1, 1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

