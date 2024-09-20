import csv
import os
import torch
import logging
import sys


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, device):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir, device):
    return get_status(model_dir, device)["vocab"]


def get_model_state(model_dir, device):
    return get_status(model_dir, device)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir, mode='w'):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, mode)
    return csv_file, csv.writer(csv_file)


def get_eval_csv_logger(model_dir, mode='w'):
    csv_path = os.path.join(model_dir, "eval.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, mode)
    return csv_file, csv.writer(csv_file)
