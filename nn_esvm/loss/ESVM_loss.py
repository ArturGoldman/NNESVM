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

    def __init__(self, lag_func="linear", bn=50, reg_lambda=0.):
        super().__init__()
        if lag_func == "linear":
            self.lagf = self.linear_w
        else:
            raise RuntimeError('Passed lag function was not recognised')
        self.bn = bn
        self.lam = reg_lambda

    def __call__(self, fbatch: Tensor, cvbatch: Tensor):
        """
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor
        """
        batch = fbatch
        if cvbatch is not None:
            batch = batch - cvbatch

        avg = batch.mean(dim=0)

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
        batch: [n_batch, dim]. Expected to receive points h(X) already
        :return: Empirical Spectral Variance, tensor(!!!)
        """

        batch = fbatch-cvbatch

        avg = batch.mean(dim=0)

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


    @staticmethod
    @torch.no_grad()
    def get_grad_norm(model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
