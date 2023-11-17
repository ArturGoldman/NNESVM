import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
from pyro.infer import HMC, MCMC, NUTS


def ula_step(prev_point, dist, gamma):
    step = (2*gamma)**0.5*torch.randn(prev_point.size(1))
    return prev_point + gamma*dist.grad_log(prev_point)+step


def mala_step(prev_point, dist, gamma):
    step = (2*gamma)**0.5*torch.randn(prev_point.size(1))
    new_point = prev_point + gamma*dist.grad_log(prev_point)+step
    a = dist.log_prob(new_point)-dist.log_prob(prev_point)
    b = (-((prev_point - new_point - gamma * dist.grad_log(new_point))**2).sum() +
                  ((new_point - prev_point - gamma * dist.grad_log(prev_point))**2).sum())/(4*gamma)
    a = torch.exp(a + b)
    u = torch.rand(1)
    if u <= a:
        return new_point
    return prev_point


class GenMCMC(nn.Module):
    def __init__(self, dist, mcmc_type="ULA", prop_scale=1., gamma=0.1):
        super().__init__()
        # note, that grad log is passed, which is -nabla U (note the minus)
        self.dist = dist
        self.mcmc_type = mcmc_type
        self.gamma = gamma
        self.mapping = {
            "ULA": ula_step,
            "MALA": mala_step
        }
        self.prop_scale = prop_scale

    def energy(self, z):
        z = z["points"]
        if z.dim() == 1:
            return -self.dist.log_prob(z.unsqueeze(0))
        return -self.dist.log_prob(z)

    def generate_chains_pyro(self, n_burn, n_clean, T, rseed=926):
        if rseed is not None:
            torch.manual_seed(rseed)

        if self.mcmc_type == "HMC":
            kernel = HMC(potential_fn=self.energy, step_size=self.gamma,
                         num_steps=5, adapt_step_size=True, full_mass=False)
        elif self.mcmc_type == "NUTS":
            kernel = NUTS(potential_fn=self.energy, full_mass=False)
        else:
            raise ValueError("MCMC sampler is not recognised")
        nbcores = multiprocessing.cpu_count()
        true_t = min(nbcores-1, T)
        if true_t < T:
            all_chains = []
            for i in range(T//true_t):
                start_points = self.prop_scale * torch.randn((true_t, self.dist.dim))
                init_params = {"points": start_points}
                mcmc = MCMC(kernel, num_samples=n_clean, warmup_steps=n_burn,
                            num_chains=true_t, initial_params=init_params, mp_context='spawn')
                mcmc.run()
                chains = mcmc.get_samples(group_by_chain=True)
                if true_t > 1:
                    all_chains.append(chains["points"].squeeze())
                else:
                    all_chains.append(chains["points"].squeeze(2))

            start_points = torch.randn((T%true_t, self.dist.dim))
            init_params = {"points": start_points}
            mcmc = MCMC(kernel, num_samples=n_clean, warmup_steps=n_burn,
                        num_chains=T%true_t, initial_params=init_params, mp_context='spawn')
            mcmc.run()
            chains = mcmc.get_samples(group_by_chain=True)
            if T%true_t > 1:
                all_chains.append(chains["points"].squeeze())
            else:
                all_chains.append(chains["points"].squeeze(2))
            return torch.cat(all_chains, dim=0)
        else:
            start_points = self.prop_scale * torch.randn((T, self.dist.dim))
            init_params = {"points": start_points}
            mcmc = MCMC(kernel, num_samples=n_clean, warmup_steps=n_burn,
                        num_chains=T, initial_params=init_params, mp_context='spawn')
            mcmc.run()
            chains = mcmc.get_samples(group_by_chain=True)

            # squeeze seems to work ok here: if chain is one, it removes redundant dimension, otherwise leaves it
            return chains["points"].squeeze()

    def gen_samples(self, n_burn, n_clear, rseed=None):
        """
        Generates one chain
        :param n_burn:
        :param n_clear:
        :param rseed:
        :return: [n_clear, dim]
        """
        if self.mcmc_type not in self.mapping:
            return self.generate_chains_pyro(n_burn, n_clear, 1, rseed=rseed)
        if rseed is not None:
            torch.manual_seed(rseed)
        prev_point = self.prop_scale * torch.randn(self.dist.dim).reshape(1, -1)
        samples = [prev_point]
        # start from 1, because we already have one starting point
        for i in tqdm(range(1, n_burn+n_clear), desc="Generating samples"):
            new_point = self.mapping[self.mcmc_type](prev_point, self.dist, self.gamma)
            samples.append(new_point)
            prev_point = new_point
        return torch.cat(samples, dim=0)[n_burn:]

    def generate_parallel_chains(self, n_burn, n_clear, T, rseed=926):
        """
        Generates multiple chains
        :param n_burn:
        :param n_clear:
        :param T:
        :param rseed:
        :return: [T, n_clear, dim]
        """
        if self.mcmc_type not in self.mapping:
            return self.generate_chains_pyro(n_burn, n_clear, T, rseed=rseed)
        nbcores = multiprocessing.cpu_count()
        ctx = multiprocessing.get_context('spawn')
        print("Total cores for multiprocessing", nbcores)
        multi = ctx.Pool(nbcores)
        res = multi.starmap(self.gen_samples,
                            [(n_burn, n_clear,
                              rseed + i) for i in range(T)])
        return torch.stack(res, dim=0)

