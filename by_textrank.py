import os.path as osp

import pandas as pd
import pytextrank
import spacy
from tqdm import tqdm

import utils


def get_keyword(doc):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    nlp_doc = nlp(doc)
    keyword = [(k.text, k.rank) for k in nlp_doc._.phrases if len(k.text.split()) <= 3]
    return keyword


def extract(dir_path, file_name):
    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="[TextRank]"):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"

        df.at[idx, "kwrd_ttl"] = get_keyword(title)
        df.at[idx, "kwrd_abs"] = get_keyword(abstract)
        df.at[idx, "kwrd_all"] = get_keyword(title_abstract)

    # Save dataframe
    save_file_name = f"{file_name}_by_textrank.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
