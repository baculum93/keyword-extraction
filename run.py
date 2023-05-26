import argparse

import by_keybert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--load_fn", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    args = parser.parse_args()

    if "keybert_default" in args.method:
        by_keybert.extract_by_default(args.dir_path, args.load_fn)

    if "keybert_nogram" in args.method:
        by_keybert.extract_with_KeyphraseCountVectorizer(args.dir_path, args.load_fn)


if __name__ == "__main__":
    main()
