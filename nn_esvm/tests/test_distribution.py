import numpy as np
import torch
from nn_esvm.distributions.distributions import BananaShape, GMM, Funnel, LogReg
import matplotlib.pyplot as plt
from nn_esvm.MCMC import GenMCMC


def plot_samples(cur, n_samples=10**3):
    samples = cur.sample(n_samples)
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()

def plot_dist(cur):
    xlim = [-2, 3]
    ylim = [-6, 6]

    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    points = []
    fval = []
    for xx in x:
        for yy in y:
            points.append([xx, yy])
            fval.append(cur.log_prob(torch.tensor([[xx, yy]])))
    points = np.array(points)
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(points[:, 0], points[:, 1], c=torch.stack(fval, dim=0))
    plt.colorbar(sc)
    plt.xlabel("X")
    plt.ylabel("Y")
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

def check_grad(cur):
    A = torch.arange(0, 16).reshape(-1, 2).float()
    print(A)
    print(cur.grad_log(A))
    print(cur.grad_log(A).size())

def check_MC(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1):
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.gen_samples(n_burn, n_clean, rseed=926)
    plt.figure(figsize=(12, 8))
    #plt.plot(samples[:, 0], samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()

def check_MC_sparse(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10):
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.gen_samples(n_burn, n_clean)
    # samples = samples[n_burn:]
    plt.figure(figsize=(12, 8))
    #plt.plot(samples[:, 0], samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()
    samples = samples[::step]
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()

def check_bias(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10, n_exp=100):
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.generate_parallel_chains(n_burn, n_clean, n_exp, rseed=926)
    print(samples.size())

    """
    samples = []
    for _ in range(n_exp):
        samples.append(cur.sample(n_clean))
    """

    samples = samples[:, ::step, :]

    for i in range(15):
        plt.scatter(samples[i, :, 0], samples[i, :, 1], c=cur.log_prob(samples[i]))
        plt.grid()
        plt.title("n_clean: {}, Mean f: {}".format(n_clean, samples[i, :, 1].mean().item()))
        plt.show()

    samples = samples[:, :, 1:2]
    print(samples.size())
    means = samples.mean(dim=(-2, -1))
    plt.figure(figsize=(12, 8))
    plt.boxplot([means], showfliers=False)
    plt.title("Clean: {}, step: {}".format(n_clean, step))
    plt.grid()
    plt.show()

def see_chains(file_name, cur):
    checkpoint = torch.load(file_name)
    chains = checkpoint["chains"]
    print(chains.size())
    for i in range(chains.size(0)):
        samples = chains[i]
        plt.figure(figsize=(12, 8))
        plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
        plt.grid()
        plt.show()


def run_tests(dist, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10):
    plot_samples(dist, 10**4)
    #plot_dist(dist)
    #check_potential(dist)
    #check_log(dist)
    #check_grad(dist)
    #check_MC(dist, n_burn, n_clean, mc_type, gamma)
    #check_MC_sparse(dist, n_burn, n_clean, mc_type, gamma, step)
    #check_bias(dist, n_burn, n_clean, mc_type, gamma, step)


if __name__ == "__main__":
    #dist = LogReg(15, "../../saved/eyes.csv", 10)
    dist = Funnel(2, a=1, b=0.5)
    plot_samples(dist, 10**4)
    #dist = Funnel(30, a=1, b=0.5)
    #dist = BananaShape(2, p=100)
    #dist = GMM(2, 1)
    #run_tests(dist, 10**5, 10**6, "MALA", 0.1, 1)
    #see_chains('../../saved/data/Funnel_30_NUTS_926.pth', dist)

