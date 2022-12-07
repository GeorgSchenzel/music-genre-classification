import argparse
from mgclass import some_module


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A nontrivial modular command")
    subparsers = parser.add_subparsers(help="sub-command help")

    add_some_subparser(subparsers)

    return parser


def add_some_subparser(subparsers):
    parser = subparsers.add_parser("something", help="Do something")

    parser.set_defaults(func=some_module.main)
    parser.add_argument("path", help="The path to the thing")


if __name__ == "__main__":
    args = create_parser().parse_args()

    args.func(args)
