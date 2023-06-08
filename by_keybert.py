"""KeyBERT

https://maartengr.github.io/KeyBERT/index.html
"""

import os.path as osp

import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from tqdm import tqdm
from transformers.pipelines import pipeline

import utils


def set_model_config(hf_model_name=None):
    config = {}
    if not hf_model_name:
        config["model_name"] = "all-MiniLM-L6-v2"
        config["hf_model"] = None
        config["kw_model"] = KeyBERT(model=config["model_name"])
    else:
        config["model_name"] = hf_model_name  # --> https://huggingface.co/models
        config["hf_model"] = pipeline(
            model=config["model_name"], task="feature-extraction"
        )
        config["kw_model"] = KeyBERT(model=config["hf_model"])
    config["top_n"] = 10
    config["use_maxsum"] = False  # diversification parameter
    # config["nr_candidates"]
    config["use_mmr"] = False  # diversification parameter
    # config["diversity"]
    config["stop_words"] = None
    return config


def extract_by_default(dir_path, file_name):
    def _get_keyword(doc):
        keyword = config["kw_model"].extract_keywords(
            docs=doc,
            stop_words=config["stop_words"],
            top_n=config["top_n"],
            use_maxsum=config["use_maxsum"],
            use_mmr=config["use_mmr"],
            keyphrase_ngram_range=(ngram_n, ngram_n),
        )
        return keyword

    logger = utils.set_logger("keybert_default")

    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    # Create new columns
    kwrd_columns = [col for col in df.columns if col.startswith("kwrd")]
    df = df.drop(columns=kwrd_columns)
    for col_prefix in kwrd_columns:
        for ngram in range(1, 4):
            df[f"{col_prefix}_{ngram}"] = None

    # Set model config
    config = set_model_config()

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="[Keybert - Default]"):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"

        for ngram_n in range(1, 4):
            # title
            try:
                df.at[idx, f"kwrd_ttl_{ngram_n}"] = _get_keyword(title)
            except Exception as e:
                df.at[idx, f"kwrd_ttl_{ngram_n}"] = []
                logger.exception(f"--> {title}")

            # abstract
            try:
                df.at[idx, f"kwrd_abs_{ngram_n}"] = _get_keyword(abstract)
            except Exception as e:
                df.at[idx, f"kwrd_abs_{ngram_n}"] = []
                logger.exception(f"--> {abstract}")

            # title + abstract
            try:
                df.at[idx, f"kwrd_all_{ngram_n}"] = _get_keyword(title_abstract)
            except Exception as e:
                df.at[idx, f"kwrd_all_{ngram_n}"] = []
                logger.exception(f"--> {title_abstract}")

    # Config Dataframe
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_keybert_default.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")


def extract_with_KeyphraseCountVectorizer(dir_path, file_name):
    def _get_keyword(doc):
        keyword = config["kw_model"].extract_keywords(
            docs=doc,
            stop_words=config["stop_words"],
            top_n=config["top_n"],
            use_maxsum=config["use_maxsum"],
            use_mmr=config["use_mmr"],
            vectorizer=vectorizer,
        )
        return keyword

    logger = utils.set_logger("keybert_nogram")

    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    # Set model config
    config = set_model_config()
    vectorizer = KeyphraseCountVectorizer(pos_pattern="<N.*>{1, 3}")

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0], desc="[Keybert - No n-gram]"):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"

        # title
        try:
            df.at[idx, f"kwrd_ttl"] = _get_keyword(title)
        except Exception as e:
            df.at[idx, f"kwrd_ttl"] = []
            logger.exception(f"--> {title}")

        # abstract
        try:
            df.at[idx, f"kwrd_abs"] = _get_keyword(abstract)
        except Exception as e:
            df.at[idx, f"kwrd_abs"] = []
            logger.exception(f"--> {abstract}")

        # title + abstract
        try:
            df.at[idx, f"kwrd_all"] = _get_keyword(title_abstract)
        except Exception as e:
            df.at[idx, f"kwrd_all"] = []
            logger.exception(f"--> {title_abstract}")

    # Config Dataframe
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_keybert_KeyphraseCountVectorizer.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")
