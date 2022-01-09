from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import nn_esvm.datasets
from nn_esvm.utils.parse_config import ConfigParser


def get_dataloader(configs: ConfigParser, target_dist, split):
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

    # select batch size or batch sampler
    assert xor("batch_size" in params, "batch_sampler" in params), \
        "You must provide batch_size or batch_sampler for each split"
    if "batch_size" in params:
        bs = params["batch_size"]
        shuffle = False
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
