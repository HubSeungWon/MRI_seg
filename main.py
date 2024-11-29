#!/usr/bin/env python3

"""
Launch script for OmniDet MTL.

# usage: ./main.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import argparse
import json
import os
import shutil
from distutils.util import strtobool
from pathlib import Path

import yaml

from utils import Tupperware


def printj(dic):
    return print(json.dumps(dic, indent=4))


def collect_args() -> argparse.Namespace:
    """Set command line arguments"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', help="Config file", type=str, default=Path(__file__).parent / "data/params.yaml")
    parser.add_argument('--config', help="Config file", type=str, default=Path(__file__).parent / "./params.yaml")
    args = parser.parse_args()
    return args


def collect_tupperware() -> Tupperware:
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    # args = Tupperware(params)
    # printj(args)
    printj(params)
    # return args
    return params


def main():
    args = collect_tupperware()
    # log_path = os.path.join(args.output_directory, args.model_name)
    log_path = os.path.join(args["output_directory"], args["model_name"])

    if os.path.isdir(log_path):
        # pass
        if strtobool(input("=> Clean up the log directory?")):
            shutil.rmtree(log_path, ignore_errors=False, onerror=None)
            os.mkdir(log_path)
            print("=> Cleaned up the logs!")
        else:
            print("=> No clean up performed!")
    else:
        print(f"=> No pre-existing directories found for this experiment. \n"
              f"=> Creating a new one!")
        os.mkdir(log_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = args["cuda_visible_devices"] or "-1"


    if args["train"] == "semantic":
        model = SemanticModel(args)
        model.semantic_train()


if __name__ == "__main__":

    from train_semantic import SemanticModel

    main()
