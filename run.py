import argparse

from by_keybert import run_keybert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True)
    parser.add_argument("--load_fn", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    args = parser.parse_args()

    if "keybert" in args.method:
        run_keybert(args.dir_path, args.load_fn)


if __name__ == "__main__":
    main()
