import argparse

import by_keybert
import by_rake
import by_textrank
import by_tfidf
from preprocessing import run_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--load_fn", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    args = parser.parse_args()

    # raw text preprocssing
    run_preprocessing(args.dir_path, args.load_fn)

    # keyword extraction
    if "keybert_default" in args.method:
        by_keybert.extract_by_default(args.dir_path, args.load_fn)

    if "keybert_nogram" in args.method:
        by_keybert.extract_with_KeyphraseCountVectorizer(args.dir_path, args.load_fn)

    if "tfidf" in args.method:
        by_tfidf.extract(args.dir_path, args.load_fn)

    if "rake" in args.method:
        by_rake.extract(args.dir_path, args.load_fn)

    if "textrank" in args.method:
        by_textrank.extract(args.dir_path, args.load_fn)


if __name__ == "__main__":
    main()
