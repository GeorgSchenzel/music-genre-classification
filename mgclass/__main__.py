import argparse
from mgclass import dataset_analyzer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mgclass,",
        description="A full pipeline for music genre classification",
    )
    subparsers = parser.add_subparsers(metavar="command", required=True)

    add_some_subparser(subparsers)

    return parser


def add_some_subparser(subparsers):
    parser = subparsers.add_parser("analyze", help="analyze raw data")

    parser.set_defaults(func=dataset_analyzer.main)
    parser.add_argument("path", help="the path to the 'spotdj.json' file")


if __name__ == "__main__":
    args = create_parser().parse_args()

    args.func(args)
