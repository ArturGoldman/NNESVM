import torch
from nn_esvm.distributions.distributions import BananaShape, GMM
import matplotlib.pyplot as plt
from nn_esvm.MCMC import GenMCMC


def plot_dist(cur):
    samples = cur.sample(1000)
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.grid()
    plt.show()


def check_potential(cur):
    A = torch.arange(0, 16).reshape(-1, 2)
    print(A)
    print(cur.potential(A))


def check_log(cur):
    A = torch.arange(0, 16).reshape(-1, 2).float()
    print(A)
    print(cur.log_prob(A))

    samples = cur.sample(1000)
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()


def check_grad(cur):
    A = torch.arange(0, 16).reshape(-1, 2).float()
    print(A)
    print(cur.grad_log(A))

def check_MC(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1):
    generator = GenMCMC(cur.grad_log, mc_type, gamma)
    samples = generator.gen_samples(n_burn+n_clean, cur.dim)
    samples = samples[n_burn:]
    plt.figure(figsize=(12, 8))
    #plt.plot(samples[:, 0], samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()

def run_tests(dist):
    #plot_dist(dist)
    #check_potential(dist)
    #check_log(dist)
    #check_grad(dist)
    check_MC(dist)


if __name__ == "__main__":
    #dist = BananaShape(2)
    dist = GMM(2, 0.5)
    run_tests(dist)

