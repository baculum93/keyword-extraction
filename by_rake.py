"""RAKE (Rapid Automatic Keyword Extraction)"""


import os.path as osp

import nltk
import pandas as pd
from rake_nltk import Rake
from tqdm import tqdm

import utils


def get_keyword(doc):
    rake = Rake()
    rake.extract_keywords_from_text(doc)
    result = []
    for score, keyword in rake.get_ranked_phrases_with_scores():
        if len(keyword.split()) <= 3:
            result.append((keyword, round(score, 5)))
    keyword = list(set(result))
    keyword.sort(key=lambda x: x[1], reverse=True)
    return keyword


def extract(dir_path, file_name):
    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="[RAKE]"):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"

        df.at[idx, "kwrd_ttl"] = get_keyword(title)
        df.at[idx, "kwrd_abs"] = get_keyword(abstract)
        df.at[idx, "kwrd_all"] = get_keyword(title_abstract)

    # Save dataframe
    save_file_name = f"{file_name}_by_rake.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
