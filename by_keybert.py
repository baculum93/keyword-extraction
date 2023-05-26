import os.path as osp

import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transformers.pipelines import pipeline
from tqdm import tqdm


HF_MODEL = pipeline("feature-extraction", model="distilbert-base-cased")
KW_MODEL = KeyBERT(model=HF_MODEL)
USE_MAXSUM = False
USE_MMR = True


def extract_by_default(dir_path, file_name):
    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        for y in range(3):
            df[f"kwrd_{x}_{y+1}"] = None

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        idx = row.Index
        title = row.preprocessed_title
        abstract = row.preprocessed_abstract

        docs = [title, abstract, f"{title}. {abstract}"]
        for i, doc in enumerate(docs):
            for n in range(1, 4):
                try:
                    keyword = KW_MODEL.extract_keywords(
                        docs=doc,
                        stop_words=None,
                        use_maxsum=USE_MAXSUM,
                        use_mmr=USE_MMR,
                        keyphrase_ngram_range=(1, n),
                    )
                    df.at[idx, f"kwrd_{text_type[i]}_{n}"] = keyword
                except Exception as e:
                    df.at[idx, f"kwrd_{text_type[i]}_{n}"] = []
                    print(e)
                    print(doc)
    
    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_by_keybert_default.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def extract_with_KeyphraseCountVectorizer(dir_path, file_name):
    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Create new columns
    text_type = {0: "ttl", 1: "abs", 2: "all"}
    for x in text_type.values():
        df[f"kwrd_{x}"] = None

    vectorizer = KeyphraseCountVectorizer(pos_pattern="<N.*>{1, 3}")

    # Extract keywords
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        idx = row.Index
        title = row.preprocessed_title
        abstract = row.preprocessed_abstract
    
        docs = [title, abstract, f"{title}. {abstract}"]
        for i, doc in enumerate(docs):
            try:
                keyword = KW_MODEL.extract_keywords(
                    docs=doc,
                    stop_words=None,
                    use_maxsum=USE_MAXSUM,
                    use_mmr=USE_MMR,
                    vectorizer=vectorizer,
                )
                df.at[idx, f"kwrd_{text_type[i]}"] = keyword
            except Exception as e:
                df.at[idx, f"kwrd_{text_type[i]}"] = []
                print(e)
                print(doc)

    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_by_keybert_KeyphraseCountVectorizer.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
