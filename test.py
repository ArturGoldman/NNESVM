import argparse
from pathlib import Path
import os
import shutil

import torch
from tqdm import tqdm

import torchaudio

import hw_nv.model as module_arch
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.aligner import Batch
from hw_nv.processing import MelSpectrogram

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config):
    logger = config.get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)
    logger.info(sum(p.numel() for p in model.parameters()))

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict_g"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    save_path = Path('./audios')
    if save_path.exists():
        shutil.rmtree(str(save_path), ignore_errors=True)
    save_path.mkdir(parents=True, exist_ok=False)

    to_test = config["file_dir"]
    fnames = os.listdir(to_test)
    melspectr = MelSpectrogram(config).to(device)
    for f in tqdm(fnames):
        waveform, old_sr = torchaudio.load(to_test+'/'+f)
        waveform = torchaudio.transforms.Resample(old_sr, 22050)(waveform).to(device)
        melspec = melspectr(waveform)
        out = model(melspec)
        torchaudio.save(str(save_path)+'/'+f, out.cpu().squeeze(0), sample_rate=22050)

    print("Testing: DONE")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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

    config = ConfigParser.from_args(args)

    main(config)