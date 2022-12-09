import json
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import mutagen


class MusicGenreDataset(Dataset):
    def __init__(
        self, data_dir: Path, transform=None, target_transform=None, num_classes=10
    ):
        self.data_dir = data_dir

        with open(data_dir / "spotdj.json", "r") as f:
            self.spotdj_data = json.load(f).get("songs")

        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes

        self.genres = self.aggregate_best_genres()
        self.files, self.labels = self.create_dataset()

    def create_dataset(self) -> (List[Path], List[int]):
        files = []
        labels = []

        genre_to_label = {genre: i for i, genre in enumerate(self.genres)}

        for datapoint in self.spotdj_data.values():
            try:
                song_file = self.data_dir / datapoint["filename"]
                mutagen_file = mutagen.File(song_file, easy=True)

                if "genre" not in mutagen_file or len(mutagen_file["genre"]) == 0:
                    continue

                genre = mutagen_file["genre"][0]
                if genre not in self.genres:
                    continue

                labels.append(genre_to_label[genre])
                files.append(song_file)

            except FileNotFoundError:
                pass

        return files, labels

    def aggregate_best_genres(self) -> List[str]:
        spotify_genres = []
        for datapoint in self.spotdj_data.values():
            try:
                mutagen_file = mutagen.File(
                    self.data_dir / datapoint["filename"], easy=True
                )

                if "genre" in mutagen_file and len(mutagen_file["genre"]) > 0:
                    spotify_genres.append(mutagen_file["genre"][0])
            except FileNotFoundError:
                pass

        return (
            pd.Series(spotify_genres)
            .value_counts(sort=True)
            .head(self.num_classes)
            .index.to_list()
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.files[idx])

        tensor = waveform
        label = self.labels[idx]

        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)

        return tensor, label
