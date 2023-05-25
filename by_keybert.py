import os.path as osp

import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from tqdm import tqdm

KW_MODEL = KeyBERT(model="all-MiniLM-L6-v2")
USE_MAXSUM = False
USE_MMR = True


def extract_by_default(*docs):
    keyword_list = []
    for doc in docs:
        temp_keyword_list = []
        for n in range(1, 4):
            try:
                keyword = KW_MODEL.extract_keywords(
                    docs=doc,
                    stop_words=None,
                    use_maxsum=USE_MAXSUM,
                    use_mmr=USE_MMR,
                    keyphrase_ngram_range=(1, n),
                )
                temp_keyword_list.append(keyword)
            except Exception as e:
                temp_keyword_list.append([])
                print(e)
                print(doc)
        keyword_list.append(temp_keyword_list)
    return keyword_list


def extract_with_KeyphraseCountVectorizer(*docs):
    vectorizer = KeyphraseCountVectorizer(pos_pattern="<N.*>{1, 2}")

    keyword_list = []
    for doc in docs:
        try:
            keyword = KW_MODEL.extract_keywords(
                docs=doc,
                stop_words=None,
                use_maxsum=USE_MAXSUM,
                use_mmr=USE_MMR,
                vectorizer=vectorizer,
            )
            keyword_list.append(keyword)
        except Exception as e:
            keyword_list.append([])
            print(e)
            print(doc)
    return keyword_list


def run_keybert(dir_path, file_name):
    # Load data
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    # Extract keywords
    df["kwrd_ttl_dflt_1"] = None
    df["kwrd_ttl_dflt_2"] = None
    df["kwrd_ttl_dflt_3"] = None
    df["kwrd_abs_dflt_1"] = None
    df["kwrd_abs_dflt_2"] = None
    df["kwrd_abs_dflt_3"] = None
    df["kwrd_all_dflt_1"] = None
    df["kwrd_all_dflt_2"] = None
    df["kwrd_all_dflt_3"] = None
    df["kwrd_ttl_kcv"] = None
    df["kwrd_abs_kcv"] = None
    df["kwrd_all_kcv"] = None
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        idx = row.Index
        title = row.preprocessed_title
        abstract = row.preprocessed_abstract

        # default
        keyword_dflt = extract_by_default(title, abstract, f"{title}. {abstract}")
        df.at[idx, "kwrd_ttl_dflt_1"] = keyword_dflt[0][0]
        df.at[idx, "kwrd_ttl_dflt_2"] = keyword_dflt[0][1]
        df.at[idx, "kwrd_ttl_dflt_3"] = keyword_dflt[0][2]
        df.at[idx, "kwrd_abs_dflt_1"] = keyword_dflt[1][0]
        df.at[idx, "kwrd_abs_dflt_2"] = keyword_dflt[1][1]
        df.at[idx, "kwrd_abs_dflt_3"] = keyword_dflt[1][2]
        df.at[idx, "kwrd_all_dflt_1"] = keyword_dflt[2][0]
        df.at[idx, "kwrd_all_dflt_2"] = keyword_dflt[2][1]
        df.at[idx, "kwrd_all_dflt_3"] = keyword_dflt[2][2]

        # KeyphraseCountVectorizer
        keyword_kcv = extract_with_KeyphraseCountVectorizer(
            title, abstract, f"{title}. {abstract}"
        )
        df.at[idx, "kwrd_ttl_kcv"] = keyword_kcv[0]
        df.at[idx, "kwrd_abs_kcv"] = keyword_kcv[1]
        df.at[idx, "kwrd_all_kcv"] = keyword_kcv[2]

    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_by_keybert.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
