# Music Genre Classification

A pipeline for training a neural network for music genre classification.

## Docker

You can run the application using docker:

```shell
$ docker pull ghcr.io/georgschenzel/music-genre-classification:latest
$ docker run -d -p 8000:8000 ghcr.io/georgschenzel/music-genre-classification:latest
```

## Requirements

- python3
- ffmpeg (for converting files)
- make (for ease of use)

## Installation

A makefile provides various utilities to install and run the program.

`make venv` creates a virtual environment and installs all requirements

`make docs-server` starts the docs server

## Usage

This project is best used as a pyhton package. This is demonstrated in the jupyter notebooks under `./experiments/`. The output of these notebooks can also be viewed on the docs page.

## Documentation

Run `make docs-server` and access the local server to read my documentation and project summary.
If this should somehow fail, then the markdown files can still be seen under `./docs/project_summary.md` and the jupyter notebooks under `./experiments/` show all my experiments.