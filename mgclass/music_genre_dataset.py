import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import mutagen
import torchaudio
from tqdm import tqdm
import os
import platform

from mgclass.timer import Timer


class MusicGenreDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        preprocess=None,
        transform=None,
        target_transform=None,
        file_transform=None,
        num_classes=10,
        dry_run=False
    ):
        self.data_dir = data_dir

        with open(data_dir / "spotdj.json", "r") as f:
            self.spotdj_data = json.load(f).get("songs")

        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        self.file_transform = file_transform
        self.num_classes = num_classes

        self.genres = self.aggregate_best_genres()

        with Timer("Dataset creation"):
            if dry_run:
                self.data, self.labels = self.create__dry_run_dataset()
            else:
                self.data, self.labels = self.create_dataset()

    def create_dataset(self) -> (List[Path], List[int]):
        data = []
        labels = []

        genre_to_label = {genre: i for i, genre in enumerate(self.genres)}

        for datapoint in tqdm(self.spotdj_data.values(), desc="Creating Dataset"):
            try:
                song_file = self.data_dir / datapoint["filename"]
                original_file = song_file

                if self.file_transform is not None:
                    song_file = self.file_transform(song_file)

                # we need to look into the original mp3 for the metadata
                # somehow mutagen can't read wav metadata
                mutagen_file = mutagen.File(original_file, easy=True)

                if "genre" not in mutagen_file or len(mutagen_file["genre"]) == 0:
                    continue

                genre = mutagen_file["genre"][0]
                if genre not in self.genres:
                    continue

                labels.append(genre_to_label[genre])

                d, sample_rate = torchaudio.load(song_file)

                if self.preprocess:
                    d = self.preprocess(d)

                data.append(d)

            except FileNotFoundError:
                pass

        return data, labels

    def create__dry_run_dataset(self) -> (List[Path], List[int]):
        data = []
        labels = []

        for l in range(self.num_classes):
            for i in range(10):
                labels.append(l)

                # random 60s data
                d = torch.rand((1, 16000 * 60))
                if self.preprocess:
                    d = self.preprocess(d)
                data.append(d)

        return data, labels

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
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label

    @staticmethod
    def ensure_enough_memory():

        if platform.system() != "Linux":
            return

        free_memory = int(os.popen("free -m").readlines()[1].split()[-1])

        if free_memory < 500:
            raise MemoryError(
                "Aborting. Less than 500mb available memory left on device. "
            )


class RepeatedLoader:
    def __init__(self, loader: DataLoader, repeat_count: int):
        self.loader = loader
        self.repeat_count = repeat_count

    def __iter__(self):
        for _ in range(self.repeat_count):
            yield from self.loader

    def __len__(self):
        return self.repeat_count * len(self.loader)
