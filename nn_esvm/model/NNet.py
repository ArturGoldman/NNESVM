import torch
import torch.nn as nn


class DummyLinear(nn.Module):
    def __init__(self, dim=2, out_dim=1, poly_deg=4):
        """
        Simple regression model, which outputs scalar product with some weight. Bias is added by default with nn.Linear layer
        :param dim: input dimension
        :param out_dim: output dimension
        :param poly_deg: adds polynomes of each coordinate to the vector up to the power poly_deg, poly_deg included. set 1 to do nothing
        """
        super().__init__()
        self.dim = dim
        self.register_buffer('poly_deg', torch.arange(1,poly_deg+1))

        self.tail = nn.Sequential(
            nn.Linear(dim*poly_deg, out_dim) # by default Linear layer adds and learns bias term, thus we have x**0
        )

    def forward(self, x):
        out = self.tail((x.unsqueeze(-1) ** self.poly_deg).reshape(1,-1))
        return out


class MLP(nn.Module):
    def __init__(self, dim=2, out_dim=1, pos_encoding_dim=0, hidden=10, blocks=3, activation="LeakyReLU"):
        """
        Multi-layer perceptron model
        :param dim: dimension of input points
        :param out_dim: output dimension
        :param pos_encoding_dim: if 0 - does nothing. for 1 and more adds positional encoding as in transformer model.
        :param hidden: width of each hidden layer
        :param blocks: number of blocks (number of hidden layers is one less).
        note that this parameter is at least 2 (at least 1 hidden layer, otherwise use dummy linear)
        :param activation: activation function. see which ones are available in code
        """
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
        elif activation == "ReCU":
            self.activation = nn.ReLU
            self.activation_power = 3
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
        self.register_buffer('mean_diff', torch.zeros(self.dim,))
        # self.head.apply(self.init_weights)
        # self.body.apply(self.init_weights)
        # self.tail.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.normal_(m.bias)
            # m.bias.data.fill_(0.01)

    def update_mean_diff(self, m):
        self.mean_diff = m
        self.mean_diff = self.mean_diff.detach()


    def forward(self, x):
        if self.ped > 0:
            add = torch.repeat_interleave(x-self.mean_diff, self.ped, dim=1) * self.freqs[None, :]
            x = torch.cat([x-self.mean_diff, torch.sin(add), torch.cos(add)], dim=1)
        else:
            x = x - self.mean_diff
        out = x
        for i in range(0, len(self.body), 2):
            out = self.body[i](out)
            out = self.body[i+1](out)
            out = out**self.activation_power
        out = self.tail(out)
        return out


