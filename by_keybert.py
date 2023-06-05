"""KeyBERT

https://maartengr.github.io/KeyBERT/index.html
"""

import os.path as osp

import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from tqdm import tqdm
from transformers.pipelines import pipeline


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
    file_name = f"{file_name}_preprocessed"

    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        for y in range(3):
            df[f"kwrd_{x}_{y+1}"] = None

    # Set model config
    config = set_model_config()

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract

        docs = [title, abstract, f"{title}. {abstract}"]
        for i, doc in enumerate(docs):
            for n in range(1, 4):
                try:
                    keyword = config["kw_model"].extract_keywords(
                        docs=doc,
                        stop_words=config["stop_words"],
                        top_n=config["top_n"],
                        use_maxsum=config["use_maxsum"],
                        use_mmr=config["use_mmr"],
                        keyphrase_ngram_range=(1, n),
                    )
                    df.at[idx, f"kwrd_{text_type[i]}_{n}"] = keyword
                except Exception as e:
                    df.at[idx, f"kwrd_{text_type[i]}_{n}"] = []
                    print(e)
                    print(doc)

    # Config Dataframe
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_keybert_default_model-{config['model_name']}_maxsum-{config['use_maxsum']}_mmr-{config['use_mmr']}.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")


def extract_with_KeyphraseCountVectorizer(dir_path, file_name):
    file_name = f"{file_name}_preprocessed"

    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        df[f"kwrd_{x}"] = None

    # Set model config
    config = set_model_config()
    vectorizer = KeyphraseCountVectorizer(pos_pattern="<N.*>{1, 3}")

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract

        docs = [title, abstract, f"{title}. {abstract}"]
        for i, doc in enumerate(docs):
            try:
                keyword = config["kw_model"].extract_keywords(
                    docs=doc,
                    stop_words=config["stop_words"],
                    top_n=config["top_n"],
                    use_maxsum=config["use_maxsum"],
                    use_mmr=config["use_mmr"],
                    vectorizer=vectorizer,
                )
                df.at[idx, f"kwrd_{text_type[i]}"] = keyword
            except Exception as e:
                df.at[idx, f"kwrd_{text_type[i]}"] = []
                print(e)
                print(doc)

    # Config Dataframe
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_keybert_KeyphraseCountVectorizer_model-{config['model_name']}_maxsum-{config['use_maxsum']}_mmr-{config['use_mmr']}.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")
