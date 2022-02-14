from torch import Tensor

# functions have to return tensor [n_batch, function_dim]


def cubic(x: Tensor):
    return (x**3).sum(dim=1, keepdim=True)


def sec_coord(x: Tensor):
    return x[:, 1].reshape(-1, 1)


def sec_coord_square(x: Tensor):
    return (x[:, 1]**2).reshape(-1, 1)
