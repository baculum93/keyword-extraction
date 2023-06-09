import os.path as osp

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import utils


def set_model_config():
    config = {}
    config["top_n"] = 10
    return config


def get_keyword(doc, word_list, vectorizer, config):
    tf_idf_vector = vectorizer.transform([doc])

    # Sort with highest score
    coo_matrix = tf_idf_vector.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    # Get word & tf-idf score
    sorted_items = sorted_items[: config["top_n"]]
    output = {}
    for idx, score in sorted_items:
        output[word_list[idx]] = round(score, 5)

    return list(output.items())


def extract(dir_path, file_name):
    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    # Set model config
    config = set_model_config()

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=(1, 1))
    vectorizer.fit_transform(df["ppd_title"] + ". " + df["ppd_abstract"])
    word_list = vectorizer.get_feature_names_out()

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="[TF-IDF]"):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"

        df.at[idx, "kwrd_ttl"] = get_keyword(title, word_list, vectorizer, config)
        df.at[idx, "kwrd_abs"] = get_keyword(abstract, word_list, vectorizer, config)
        df.at[idx, "kwrd_all"] = get_keyword(
            title_abstract, word_list, vectorizer, config
        )

    # Config Dataframe
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_tfidf.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")
