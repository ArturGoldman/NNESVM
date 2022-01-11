from torch import nn, Tensor
import torch
from tqdm import tqdm


class LossESVM(nn.Module):
    @staticmethod
    def linear_w(s):
        if abs(s) > 1:
            raise ValueError('Lag function got unexpected value')
        if 0.5 < abs(s) <= 1:
            return -2*abs(s)+2
        return 1

    def __init__(self, lag_func="linear", bn=50):
        super().__init__()
        if lag_func == "linear":
            self.lagf = self.linear_w
        else:
            raise RuntimeError('Passed lag function was not recognised')
        self.bn = bn

    def __call__(self, batch: Tensor):
        """
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor
        """

        avg = batch.mean(dim=0)
        n = batch.size(0)
        loss = ((batch-avg)**2).sum() / n
        for s in range(1, self.bn):
            loss += 2 * self.lagf(s/self.bn)*((batch[:-s]-avg)*(batch[s:]-avg)).sum()/n
        return loss


class SmartLossESVM(LossESVM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch: Tensor):
        """
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor(!!!)
        """

        avg = batch.mean(dim=0)
        n = batch.size(0)
        loss_accum = 0
        loss = ((batch-avg)**2).sum() / n
        loss_accum += loss.item()
        loss.backward(retain_graph=True)
        for s in tqdm(range(1, self.bn), desc="Calculating loss"):
            loss = 2 * self.lagf(s/self.bn)*((batch[:-s]-avg)*(batch[s:]-avg)).sum()/n
            loss_accum += loss.item()
            loss.backward(retain_graph=True)
        return torch.tensor(loss_accum)
