import argparse
import collections
import warnings

import torch
import time

from nn_esvm.utils.parse_config import ConfigParser
from nn_esvm.MCMC import GenMCMC
import nn_esvm.distributions
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


def main(config):
    logger = config.get_logger("gen_samples")

    target_distribution = config.init_obj(config["target_dist"], nn_esvm.distributions)

    chain_generator = GenMCMC(target_distribution, config["chain_details"]["mcmc_type"],
                              config["chain_details"]["gamma"])

    start = time.time()
    chains = chain_generator.generate_parallel_chains(config["chain_details"]["n_burn"],
                                                      config["chain_details"]["n_clean"],
                                                      config["chain_details"]["n_chains"],
                                                      config["chain_details"]["rseed"])
    fin = time.time()
    tot = fin-start

    state = {
        "chains": chains
    }
    save_dir = Path(config["trainer"]["save_dir"])
    sec_part = config["target_dist"]['type']+'_'+str(config["target_dist"]['args']['dim'])+\
               '_'+config["chain_details"]["mcmc_type"]+'_'+str(config["chain_details"]["rseed"])+'.pth'
    save_name = save_dir/"data"
    save_name.mkdir(parents=True, exist_ok=True)
    save_name = save_name/sec_part
    torch.save(state, save_name)
    logger.info("Chains were generated in {} hours, {} minutes, {} seconds".format(tot//3600, (tot%3600)//60, tot%60))
    logger.info("Data saved to: {}".format(str(save_name)))

    if config["to_check"]:
        checkpoint = torch.load(save_name)
        cur_chains = checkpoint["chains"]
        print(cur_chains)
        print(cur_chains.size())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
