import os.path as osp

import pandas as pd
from thefuzz import process


def preprocess_dataframe(dir_path, file_name):
    file_name = f"{file_name}_preprocessed"

    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        df[f"kwrd_{x}"] = None

    return df


def dedupe_keyword(kewords, threshold=85):
    return list(process.dedupe(kewords, threshold=threshold))
