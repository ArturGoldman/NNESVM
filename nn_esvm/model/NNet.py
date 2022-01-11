import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim=2, out_dim=1, hidden=10, blocks=3):
        super().__init__()
        # TODO:positional encoding for lower dimensions
        # is positional encoding needed though? i believe CV should be pretty smooth

        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU()
        )

        mods = []
        for i in range(blocks-2):
            mods += [
                nn.Linear(hidden, hidden),
                nn.LeakyReLU()
            ]

        self.body = nn.Sequential(*mods)

        self.tail = nn.Sequential(
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)
        return out

