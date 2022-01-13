from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import nn_esvm.datasets
from nn_esvm.utils.parse_config import ConfigParser
import matplotlib.pyplot as plt


def get_dataloader(configs: ConfigParser, target_dist, split, writer):
    params = configs["data"][split]
    num_workers = params.get("num_workers", 1)

    drop_last = False

    # create and join datasets
    datasets = []
    for ds in params["datasets"]:
        datasets.append(configs.init_obj(
                ds, nn_esvm.datasets, dist=target_dist))

    assert len(datasets)
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]
    if target_dist.dim == 2:
        ch = dataset.chain
        plt.figure(figsize=(12, 8))
        plt.scatter(ch[:, 0], ch[:, 1], c=target_dist.log_prob(ch))
        plt.grid()
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        writer.add_image("Training_chain", plt)

    # select batch size or batch sampler
    assert xor("batch_size" in params, "batch_sampler" in params), \
        "You must provide batch_size or batch_sampler for each split"
    if "batch_size" in params:
        bs = params["batch_size"]
        shuffle = params["shuffle"]
        batch_sampler = None
    else:
        raise Exception()

    # create dataloader
    dataloader = DataLoader(
        dataset, batch_size=bs,
        shuffle=shuffle, num_workers=num_workers,
        batch_sampler=batch_sampler, drop_last=drop_last
    )
    return dataloader
