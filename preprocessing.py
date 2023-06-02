import os.path as osp
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("wordnet")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 화학 기호와 숫자 사이의 공백 제거
    chem_regex = r"[A-Z]+[a-z]*\s+\d+"
    text = re.sub(chem_regex, lambda x: x.group().replace(" ", ""), text)

    # html <sub> tag 제거
    html_regex = r"<sub>(.*)<\/sub>"
    text = re.sub(html_regex, r"\1", text)

    # 2번 이상 반복된 null 제거
    null_regex = r"\bnull\b( \bnull\b)+"
    text = re.sub(null_regex, "", text)

    # underbar 제거
    text = text.replace("_", "")

    # $ 제거
    text = text.replace("$", "")

    # tokenize & stropword
    stopword = set(stopwords.words("english"))
    token = word_tokenize(text)
    word_list = [w for w in token if w not in stopword]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(w) for w in word_list]

    return " ".join(word_list)


def run_preprocessing(dir_path, file_name):
    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    df["ppd_title"] = None
    df["ppd_abstract"] = None
    for row in df.itertuples():
        df.at[row.Index, "ppd_title"] = preprocess_text(row.title)
        df.at[row.Index, "ppd_abstract"] = preprocess_text(row.abstract)

    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_preprocessed.csv")
    df.to_csv(save_path, index=False)
