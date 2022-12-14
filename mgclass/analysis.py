import json
from pathlib import Path
from typing import List

import librosa as librosa
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import mutagen

from mgclass import MusicGenreDataset


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_spectrogram2(data):
    sb.heatmap(data)
    plt.show()


def plot_frequency_chart(df, title: str, rot=True):
    # print(title)
    # print(df.to_string())

    ax = df.plot(kind="bar", rot=45 if rot else 0)

    if rot:
        ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_frequency_dataframe(data: List):
    df = pd.Series(data).value_counts(sort=True)
    return df.head(10)


def summarize_spotdj_database(path: Path):
    with open(path, "r") as openfile:
        data = json.load(openfile)

    songs = data["songs"].values()

    print(f"Dataset size: {len(songs)}")
    print(f"With genres: {len([s for s in songs if len(s['genres']) > 0])}")
    print(
        f"With genres (incl. album): {len([s for s in songs if len(s['genres']) > 0 or len(s['album_genres']) > 0])}"
    )

    # plot genres
    genre_bag = []
    spotify_genre_bag = []
    count_bag = []

    for song in songs:
        count_bag.append(len(song["genres"]))

        for genre in song["genres"]:
            genre_bag.append(genre)

        try:
            mutagen_file = mutagen.File(path.parent / song["filename"], easy=True)

            if "genre" in mutagen_file:
                for genre in mutagen_file["genre"]:
                    spotify_genre_bag.append(genre)

        except FileNotFoundError:
            pass

    df = create_frequency_dataframe(genre_bag)
    plot_frequency_chart(df, "RYM Genres")

    # plot number of genres
    df = create_frequency_dataframe(count_bag)
    plot_frequency_chart(df, "RYM Genre counts", rot=False)

    # plot number of spotify genres
    df = create_frequency_dataframe(spotify_genre_bag)
    plot_frequency_chart(df, "Spotify Genres")


def summarize_dataset(dataset: MusicGenreDataset):
    genre_bag = []

    for _, label in dataset:
        genre = dataset.genres[label]
        genre_bag.append(genre)

    df = create_frequency_dataframe(genre_bag)
    plot_frequency_chart(df, "Genres")


async def main(args):
    pass