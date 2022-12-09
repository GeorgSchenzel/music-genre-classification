import json
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
import mutagen


class DatatAnalyzer:
    def __init__(self, path: Path):

        self.dir = path.parent

        with open(path, "r") as openfile:
            self.data = json.load(openfile)

        self.plot()

    def plot(self):
        songs = self.data["songs"].values()

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
                mutagen_file = mutagen.File(self.dir / song["filename"], easy=True)

                if "genre" in mutagen_file:
                    for genre in mutagen_file["genre"]:
                        spotify_genre_bag.append(genre)

            except FileNotFoundError:
                pass

        df = self.create_frequency_dataframe(genre_bag)
        self.plot_frequency_chart(df, "RYM Genres")

        # plot number of genres
        df = self.create_frequency_dataframe(count_bag)
        self.plot_frequency_chart(df, "RYM Genre counts", rot=False)

        # plot number of spotify genres
        df = self.create_frequency_dataframe(spotify_genre_bag)
        self.plot_frequency_chart(df, "Spotify Genres")

    @staticmethod
    def plot_frequency_chart(df, title: str, rot=True):
        print(title)
        print(df.to_string())
        print()

        ax = df.plot(kind="bar", rot=45 if rot else 0)

        if rot:
            ax.set_xticklabels(ax.get_xticklabels(), ha="right")

        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_frequency_dataframe(data: List):
        df = pd.Series(data).value_counts(sort=True)
        return df.head(10)


async def main(args):
    DatatAnalyzer(Path(args.path))
