import yaml
import ffmpeg
from tqdm import tqdm

from pathlib import Path
from typing import List
from spotdj.spotdj import Spotdj
from mgclass import utils


async def download_playlists(urls: List[str], destination: Path):
    with Spotdj(destination, use_rym_metadata=False) as sdj:
        for playlist in urls:
            await sdj.download_playlist(playlist)


def to_wav(input_file: Path, output_file: Path, sample_rate=16000):
    stream = ffmpeg.input(str(input_file))
    stream = stream.output(str(output_file), ac=1, ar=sample_rate)
    stream.run(quiet=True)


def convert_all(source_dir: Path):
    glob = list(source_dir.glob("*.mp3"))
    for source_file in tqdm(glob, desc="Converting to .wav"):
        destination_file = utils.mp3_to_wav_location(source_file)
        if destination_file.exists():
            continue

        to_wav(source_file, destination_file)


async def download_command(args):
    with open(args.playlists_file, "r") as f:
        urls = yaml.safe_load(f)

    await download_playlists(urls, Path(args.destination))


async def convert_command(args):
    convert_all(Path(args.source_dir))
