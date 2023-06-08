import os.path as osp

import pandas as pd
import yake


def yake_extract_keywords(text, max_ngram_size):
    language = "en"
    deduplication_threshold = 0.8
    window_size = 4
    top_k = 10

    kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        windowsSize=window_size,
        top=top_k,
    )

    keywords = kw_extractor.extract_keywords(text)
    return keywords


def extract_by_yake(dir_path, file_name):
    df = pd.read_csv(osp.join(dir_path, f"{file_name}_preprocessed.csv"))
    df["ppd_total_text"] = df["ppd_title"] + ". " + df["ppd_abstract"]
    df["reverse_ppd_total_text"] = df["ppd_abstract"] + ". " + df["ppd_title"]
    for i in range(1, 4):
        column_head = "yake_keyword_"
        column_title = column_head + "title_" + str(i)
        column_abstract = column_head + "abstract_" + str(i)
        column_total_text = column_head + "total_text_" + str(i)
        column_total_text_reverse = column_head + "reverse_total_text_" + str(i)

        df[column_title] = None
        df[column_abstract] = None
        df[column_total_text] = None
        df[column_total_text_reverse] = None
        for row in df.itertuples():
            df.at[row.Index, column_total_text] = yake_extract_keywords(
                row.ppd_total_text, i
            )
            df.at[row.Index, column_total_text_reverse] = yake_extract_keywords(
                row.reverse_ppd_total_text, i
            )
            df.at[row.Index, column_title] = yake_extract_keywords(row.ppd_title, i)
            df.at[row.Index, column_abstract] = yake_extract_keywords(
                row.ppd_abstract, i
            )

    save_path = osp.join(dir_path, f"{file_name}_yake.csv")
    df.to_csv(save_path, index=False)
