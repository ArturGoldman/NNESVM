from torch import nn, Tensor
import torch
from tqdm import tqdm
import math


class LossESVM(nn.Module):
    @staticmethod
    def linear_w(s):
        if abs(s) > 1:
            raise ValueError('Lag function got unexpected value')
        if 0.5 < abs(s) <= 1:
            return -2*abs(s)+2
        return 1

    @staticmethod
    def triangular_w(s):
        if abs(s) > 1:
            raise ValueError('Lag function got unexpected value')
        return 1-abs(s)

    @staticmethod
    def cosine_w(s):
        if abs(s) > 1:
            raise ValueError('Lag function got unexpected value')
        return 0.5+0.5*math.cos(math.pi*abs(s))

    def __init__(self, lag_func="linear", bn=50, reg_lambda=0.):
        """
        :param lag_func: type of lag function to use
        :param bn: bn parameter
        :param reg_lambda: lambda parameter used for regulariser part. currently not used
        """
        super().__init__()
        if lag_func == "linear":
            self.lagf = self.linear_w
        elif lag_func == "triangular":
            self.lagf = self.triangular_w
        elif lag_func == "cosine":
            self.lagf = self.cosine_w
        else:
            raise RuntimeError('Passed lag function was not recognised')
        self.bn = bn
        self.lam = reg_lambda

    def __call__(self, fbatch: Tensor, cvbatch: Tensor):
        """
        Regular loss function (Empirical Spectral Variance). Can be used for metric tracking
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor
        """
        batch = fbatch
        if cvbatch is not None:
            batch = batch - cvbatch

        avg = batch.mean(dim=0)
        #avg = fbatch.mean(dim=0)

        n = batch.size(0)
        loss = ((batch-avg)**2).sum() / n
        # loss = (batch**2).sum() / n
        for s in range(1, self.bn):
            loss += 2 * self.lagf(s/self.bn)*((batch[:-s]-avg)*(batch[s:]-avg)).sum()/n
        # note that we added reg to loss, though it should not appear in metric
        # if cvbatch is not None:
        #     loss += self.lam*(cvbatch**2).mean()
        return loss


class SmartLossESVM(LossESVM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, fbatch: Tensor, cvbatch: Tensor):
        """
        "Smarter" loss function. Propagates term by term in order not to store large computation graph in memory
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor(!!!)
        """

        batch = fbatch-cvbatch

        avg = batch.mean(dim=0)
        #avg = fbatch.mean(dim=0)

        n = batch.size(0)
        loss_accum = 0
        loss = ((batch-avg)**2).sum()/n
        loss_accum += loss.item()
        loss.backward(retain_graph=True)
        for s in tqdm(range(1, self.bn), desc="Calculating loss"):
            loss = 2 * self.lagf(s/self.bn)*((batch[:-s]-avg)*(batch[s:]-avg)).sum()/n
            loss_accum += loss.item()
            loss.backward(retain_graph=True)
        loss = self.lam*(cvbatch**2).mean()
        loss_accum += loss.item()
        loss.backward()
        return torch.tensor(loss_accum)
