import json
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


class DatasetAnalyzer:
    def __init__(self, path: Path):
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
        count_bag = []

        for song in songs:
            count_bag.append(len(song["genres"]))

            for genre in song["genres"]:
                genre_bag.append(genre)

        df = pd.Series(genre_bag).value_counts(sort=True)
        print(df.to_string())

        df.plot(kind="bar")
        plt.show()

        # plot number of genres
        df = pd.Series(count_bag).value_counts(sort=True)
        print(df.to_string())

        df.plot(kind="bar")
        plt.show()


def main(args):
    DatasetAnalyzer(Path(args.path))
