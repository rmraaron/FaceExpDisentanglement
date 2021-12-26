
import logging
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from path import *


def log_table(dict_list, col_list=None, part_name=""):
    if not col_list:
        col_list = list(dict_list[0].keys() if dict_list else [])
    my_list = [col_list]
    for item in dict_list:
        my_list.append([str(item[col] if item[col] is not None else '') for col in col_list])
    colSize = [max(map(len,col)) for col in zip(*my_list)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    my_list.insert(1, ['-' * i for i in colSize])
    if part_name != "":
        length = len(formatStr.format(*my_list[0]))
        logging.info("-" * round(length / 2 - len(part_name) / 2 - 1) + " " + part_name + " " +
                     "-" * int(length / 2 - len(part_name) / 2 - 1))
    for item in my_list:
        logging.info(formatStr.format(*item))


class TrainingLogger(object):
    def __init__(self, log_name_dir, args):
        self.log_name_dir = log_name_dir
        self.args = args

        with open(self.log_name_dir + "config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # Initialise running config.
        self.running_metrics = {}

        # Initialise in-epoch training.
        self.epoch_logs = {}
        self.total_size = {}

        # Initialise Tensorboard.
        self.tfb_writer = SummaryWriter(log_dir=LOGS_PATH + "tf_board/" + args.name)
        self.hparam_dict = vars(args)

    def log_batch_loss(self, name, value, partition, size, part_name=""):
        if part_name not in self.epoch_logs:
            self.epoch_logs[part_name] = {}
        if part_name not in self.total_size:
            self.total_size[part_name] = {}

        key = name + "/" + partition
        if key in self.epoch_logs[part_name]:
            self.epoch_logs[part_name][key] += value * size  # Assume value is averaged regarding the batch.
            self.total_size[part_name][key] += size
        else:
            self.epoch_logs[part_name][key] = value * size
            self.total_size[part_name][key] = size

    def log_epoch(self, log_dict, part_name=""):
        epoch = log_dict["epoch"]
        if part_name not in self.running_metrics:
            self.running_metrics[part_name] = {"epochs_trained": 0}
        self.running_metrics[part_name]["epochs_trained"] = epoch

        # Log Console and Tensorboard.
        train_dict = {"partition": "train"}
        eval_dict = {"partition": "eval"}
        for key in self.epoch_logs[part_name]:
            self.epoch_logs[part_name][key] /= self.total_size[part_name][key]
            if "train" in key:
                train_dict[key.split("/")[0]] = "%.9f" % self.epoch_logs[part_name][key]
            if "eval" in key:
                eval_dict[key.split("/")[0]] = "%.9f" % self.epoch_logs[part_name][key]
        # logging.info("  *** " + part_name + " ***")
        log_table([train_dict, eval_dict], part_name=part_name)

        scalars = self.epoch_logs[part_name]
        for key, value in scalars.items():
            self.tfb_writer.add_scalar(key, value, epoch)

        # Reset for next epoch.
        self.epoch_logs[part_name].clear()
        self.total_size[part_name].clear()
