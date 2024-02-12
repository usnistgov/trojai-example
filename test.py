import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo")
    parser.add_argument("--bar")
    args = parser.parse_args()
    print(f"{args.foo=}")
    print(f"{args.bar=}")