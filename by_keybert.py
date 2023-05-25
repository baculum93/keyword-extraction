import os.path as osp
import pandas as pd
from tqdm import tqdm
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


def extract_by_default(*docs):
    kw_model = KeyBERT()

    keyword_list = []
    for doc in docs:
        temp_keyword_list = []
        for n in range(1, 4):
            try:
                keyword = kw_model.extract_keywords(
                        docs=doc, 
                        keyphrase_ngram_range=(1, n),
                        stop_words=None
                    )
                temp_keyword_list.append(keyword)
            except Exception as e:
                temp_keyword_list.append([])
                print(e)
                print(doc)
        keyword_list.append(temp_keyword_list)
    return keyword_list


def extract_with_KeyphraseCountVectorizer(*docs, use_maxsum=False, use_mmr=False):
    kw_model = KeyBERT()
    vectorizer = KeyphraseCountVectorizer()

    keyword_list = []
    for doc in docs:
        try:
            keyword = kw_model.extract_keywords(
                    docs=doc, 
                    stop_words=None,
                    vectorizer=vectorizer, 
                    use_maxsum=use_maxsum,
                    use_mmr=use_mmr
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
        keyword_kcv = extract_with_KeyphraseCountVectorizer(title, abstract, f"{title}. {abstract}")
        df.at[idx, "kwrd_ttl_kcv"] = keyword_kcv[0]
        df.at[idx, "kwrd_abs_kcv"] = keyword_kcv[1]
        df.at[idx, "kwrd_all_kcv"] = keyword_kcv[2]
        
    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_by_keybert.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
