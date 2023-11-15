from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import nn_esvm.datasets
import nn_esvm.collator
from nn_esvm.utils.parse_config import ConfigParser
import matplotlib.pyplot as plt


def get_dataloader(configs: ConfigParser, target_dist, split, writer):
    params = configs["data"][split]
    num_workers = params.get("num_workers", 1)

    collate_fn = None
    if "collate_fn" in params:
        collate_fn = configs.init_obj(params["collate_fn"], nn_esvm.collator)

    drop_last = True

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

    # plot two first coordinates
    if target_dist.dim > 1:
        ch = dataset.chain
        plt.figure(figsize=(12, 8))
        plt.scatter(ch[:, 0], ch[:, 1], c=target_dist.log_prob(ch))
        plt.grid()
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        writer.add_image("Training_chain, first two coords", plt)

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
    # here batch size = length of the whole chain, the collator takes care of taking the correct subsample

    dataloader = DataLoader(
        dataset, batch_size=bs,
        shuffle=shuffle, num_workers=num_workers,
        batch_sampler=batch_sampler, drop_last=drop_last,
        collate_fn=collate_fn
    )
    return dataloader
