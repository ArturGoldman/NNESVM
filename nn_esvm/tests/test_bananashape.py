import torch
from nn_esvm.distributions.BananaShape import BananaShape
import matplotlib.pyplot as plt


def plot_dist():
    cur = BananaShape(2)
    samples = cur.sample(1000)
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.grid()
    plt.show()


def check_potential():
    cur = BananaShape(2)
    A = torch.arange(0, 15).reshape(3, 5)
    print(A)
    print(cur.potential(A))


def check_grad():
    cur = BananaShape(5)
    A = torch.arange(0, 15).reshape(3, 5).float()
    print(A)
    print(cur.potential(A))
    print(cur.log_prob(A))


if __name__ == "__main__":
    #plot_dist()
    #check_potential()
    check_grad()

