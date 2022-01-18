import torch
from nn_esvm.distributions.distributions import BananaShape, GMM
import matplotlib.pyplot as plt
from nn_esvm.MCMC import GenMCMC


def plot_dist(cur, n_samples=10**3):
    samples = cur.sample(n_samples)
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
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
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.gen_samples(n_burn+n_clean, cur.dim, rseed=926)
    # samples = samples[n_burn:]
    plt.figure(figsize=(12, 8))
    #plt.plot(samples[:, 0], samples[:, 1])
    plt.scatter(samples[:, 0], samples[:, 1], c=cur.log_prob(samples))
    plt.grid()
    plt.show()

def check_MC_sparse(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10):
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.gen_samples(n_burn+n_clean, cur.dim)
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

def test_bias(cur, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10, n_exp=100):
    generator = GenMCMC(cur, mc_type, gamma)
    samples = generator.generate_parallel_chains(n_burn + n_clean, cur.dim, n_exp, rseed=926)

    """
    samples = []
    for _ in range(n_exp):
        samples.append(cur.sample(n_clean))
    """

    samples = torch.stack(samples, dim=0)
    print(samples.size())
    samples = samples[:, n_burn::step, :]

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

def run_tests(dist, n_burn=1000, n_clean=1000, mc_type="ULA", gamma=0.1, step=10):
    #plot_dist(dist, 10**2)
    #check_potential(dist)
    #check_log(dist)
    #check_grad(dist)
    #check_MC(dist, n_burn, n_clean, mc_type, gamma)
    #check_MC_sparse(dist, n_burn, n_clean, mc_type, gamma, step)
    test_bias(dist, n_burn, n_clean, mc_type, gamma, step)


if __name__ == "__main__":
    dist = BananaShape(2, p=30)
    #dist = GMM(2, 0.5)
    run_tests(dist, 10**5, 10**6, "ULA", 0.01, 1)

