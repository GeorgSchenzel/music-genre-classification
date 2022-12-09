import argparse
import asyncio

from mgclass import data_downloader, datat_analyzer, training


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mgclass,",
        description="A full pipeline for music genre classification",
    )
    subparsers = parser.add_subparsers(metavar="command", required=True)

    add_data_downloader_subparser(subparsers)
    add_data_analyzer_subparser(subparsers)
    add_dataset_generator_subparser(subparsers)

    return parser


def add_data_downloader_subparser(subparsers):
    parser = subparsers.add_parser("download", help="download raw data")

    parser.set_defaults(func=data_downloader.main)
    parser.add_argument("path", help="the directory to store the raw data in")
    parser.add_argument(
        "playlists_file",
        help="A file containing the playlists to download, must be a .yaml with urls in a list",
    )


def add_dataset_generator_subparser(subparsers):
    parser = subparsers.add_parser("train", help="train a model")

    parser.set_defaults(func=training.main)


def add_data_analyzer_subparser(subparsers):
    parser = subparsers.add_parser("analyze", help="analyze raw data")

    parser.set_defaults(func=datat_analyzer.main)
    parser.add_argument("path", help="the path to the 'spotdj.json' file")


if __name__ == "__main__":
    args = create_parser().parse_args()

    asyncio.run(args.func(args))
