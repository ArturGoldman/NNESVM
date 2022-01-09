from torch import Tensor


def cubic(x: Tensor):
    return (x**3).sum()


def sec_coord(x: Tensor):
    return x[:, 1]
