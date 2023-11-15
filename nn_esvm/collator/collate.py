import random
import torch


class CollateSubSample(object):
    def __init__(self, subsample_sz):
        self.subsample_sz = subsample_sz
        self.cnt = 0

    def __call__(self, batch):
        """
        you sequentially get every n_step sample starting from self.cnt
        after several calls you get all subsamples and you start over
        :param batch: List[torch.tensor]
        :return: modified batch
        """
        n_step = len(batch)//self.subsample_sz
        self.cnt += 1
        if self.cnt >= n_step:
            self.cnt = 0
        return torch.stack(batch[self.cnt::n_step], dim=0)

