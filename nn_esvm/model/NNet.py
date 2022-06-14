import torch
import torch.nn as nn


class DummyLinear(nn.Module):
    def __init__(self, dim=2, out_dim=1):
        super().__init__()
        self.dim = dim

        self.tail = nn.Sequential(
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        out = self.tail(x)
        return out


class MLP(nn.Module):
    def __init__(self, dim=2, out_dim=1, pos_encoding_dim=0, hidden=10, blocks=3, activation="LeakyReLU"):
        super().__init__()
        self.dim = dim
        self.activation_power = 1
        self.activation = nn.LeakyReLU
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid
        elif activation == "ReLU":
            self.activation = nn.ReLU
        elif activation == "Tanh":
            self.activation = nn.Tanh
        elif activation == "ReQU":
            self.activation = nn.ReLU
            self.activation_power = 2
        self.ped = pos_encoding_dim

        mods = [
            nn.Linear(dim + 2 * dim * pos_encoding_dim, hidden),
            self.activation()
        ]
        for i in range(blocks - 2):
            mods += [
                nn.Linear(hidden, hidden),
                self.activation()
            ]

        self.body = nn.ModuleList(mods)

        self.tail = nn.Sequential(
            nn.Linear(hidden, out_dim)
        )

        freqs = (2. ** torch.arange(0, self.ped)).repeat(self.dim)
        self.register_buffer('freqs', freqs)
        # self.head.apply(self.init_weights)
        # self.body.apply(self.init_weights)
        # self.tail.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.normal_(m.bias)
            # m.bias.data.fill_(0.01)

    def forward(self, x):
        if self.ped > 0:
            add = torch.repeat_interleave(x, self.ped, dim=1) * self.freqs[None, :]
            x = torch.cat([x, torch.sin(add), torch.cos(add)], dim=1)
        out = x
        for i in range(0, len(self.body), 2):
            out = self.body[i](out)
            out = self.body[i+1](out)
            out = out**self.activation_power
        out = self.tail(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, hidden=10, activation=nn.LeakyReLU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            activation()
        )

    def forward(self, x):
        out = x + self.net(x)
        return out


class ResNet(nn.Module):
    def __init__(self, dim=2, out_dim=1, hidden=10, blocks=3):
        super().__init__()

        self.activation = nn.LeakyReLU
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            self.activation()
        )

        mods = []
        for i in range(blocks - 2):
            mods += [
                ResBlock(hidden, self.activation)
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
