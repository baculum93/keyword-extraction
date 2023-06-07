"""RAKE (Rapid Automatic Keyword Extraction)"""


import os.path as osp

import nltk
import pandas as pd
from rake_nltk import Rake

nltk.download("punkt")


def get_keyword(doc):
    rake = Rake()
    rake.extract_keywords_from_text(doc)
    result = []
    for score, keyword in rake.get_ranked_phrases_with_scores():
        if len(keyword.split()) <= 3:
            result.append((keyword, round(score, 5)))
    result = list(set(result))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def extract(dir_path, file_name):
    file_name = f"{file_name}_preprocessed"

    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        df[f"kwrd_{x}"] = None

    # Extract keywords
    for row in df.itertuples():
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
