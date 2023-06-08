import logging
import os.path as osp

import pandas as pd
from thefuzz import process


def set_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # formatter 지정
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)7s] [%(filename)18s:%(lineno)4d] (%(funcName)20s) >> %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    save_path = f"_log/{logger_name}.log"
    file_handler = logging.FileHandler(filename=save_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def preprocess_dataframe(dir_path, file_name):
    file_name = f"{file_name}_preprocessed"

    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    col_names = ["ttl", "abs", "all"]
    for col_name in col_names:
        df[f"kwrd_{col_name}"] = None

    return df


def dedupe_keyword(kewords, threshold=85):
    return list(process.dedupe(kewords, threshold=threshold))
