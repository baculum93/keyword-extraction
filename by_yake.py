import os.path as osp

import pandas as pd
import yake
import tqdm
import utils

def set_model_config():
    config = {}
    config["lan"] = "en" # text language
    config["dedupLim"] = 0.8 # deduplication threshold
    config["windowsSize"] = 4 # number of keywords
    config["top"] = 10
    return config
    

def get_keywords(doc, ngram):
    # Set model config
    config = set_model_config()
    kw_extractor = yake.KeywordExtractor(
        lan = config["lan"], 
        n = ngram,
        dedupLim = config["dedupLim"], 
        windowsSize = config["windowsSize"],
        top = config["top"] 
        )
    
    keywords = kw_extractor.extract_keywords(doc)
    return keywords


def extract(dir_path, file_name):
    # Load data
    df = utils.preprocess_dataframe(dir_path, file_name)

    for row in tqdm(df.itertuples()):
        idx = row.Index
        title = row.ppd_title
        abstract = row.ppd_abstract
        title_abstract = f"{title}. {abstract}"
        # Extract keyword
        for ngram in range(1,4):
            df.at[idx, "kwrd_ttl"] = get_keywords(title, ngram)
            df.at[idx, "kwrd_abs"] = get_keywords(abstract, ngram)
            df.at[idx, "kwrd_all"] = get_keywords(title_abstract, ngram)

    # Config Dataframe
    config = set_model_config()
    config_df = pd.DataFrame(data=config, index=[0])
    config_df = config_df.T

    # Save dataframe
    save_file_name = f"{file_name}_by_yake.xlsx"
    save_file_path = osp.join(dir_path, save_file_name)
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keyword")
        config_df.to_excel(writer, sheet_name="config")