import json

from pathlib import Path
from random import shuffle
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import mutagen
import torchaudio
from tqdm.autonotebook import tqdm
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
        dry_run=False,
        playlist_to_genre: Dict[str, str] = None,  # should be playlistid: genre label,
        max_frames=-1,
        even_classes=True,
    ):
        self.data_dir = data_dir

        with open(data_dir / "spotdj.json", "r") as f:
            j = json.load(f)
            self.spotdj_data = j.get("songs")
            self.playlist_data = j.get("playlists")

        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        self.file_transform = file_transform
        self.max_frames = max_frames
        self.even_classes = even_classes

        # Note: this is only usable when using genres from playlists and even_classes = True
        self.class_size = None

        with Timer("Dataset creation"):

            # Various ways of creating a dataset are provided
            # A dry_run option is used for fast iteration to try out training code
            if dry_run:
                self.genres = sorted(self.aggregate_best_genres(num_classes))
                self.num_classes = num_classes
                self.data, self.labels = self.create_dry_run_dataset()

            # when this dict is provided we assign each song
            # the label based on from which playlist it originates from
            elif playlist_to_genre is not None:
                print(f"Using genre from playlist source")
                self.genres = sorted(set(playlist_to_genre.values()))
                self.num_classes = len(self.genres)
                self.data, self.labels = self.create_dataset_from_playlist_genre(
                    playlist_to_genre
                )

            # as default we use the mp3 metadata to extrac genre information
            else:
                print(f"Using most {num_classes} occurring genres from spotify api")
                self.genres = sorted(self.aggregate_best_genres(num_classes))
                self.num_classes = num_classes
                self.data, self.labels = self.create_dataset_from_id3_genre()

    def create_dataset_from_id3_genre(self) -> (List[Path], List[int]):
        files = []
        labels = []

        genre_to_label = {genre: i for i, genre in enumerate(self.genres)}

        for datapoint in self.spotdj_data.values():
            song_file = self.data_dir / datapoint["filename"]

            # we need to look into the original mp3 for the metadata
            # somehow mutagen can't read wav metadata
            try:
                mutagen_file = mutagen.File(song_file, easy=True)
            except FileNotFoundError:
                continue

            if "genre" not in mutagen_file or len(mutagen_file["genre"]) == 0:
                continue

            genre = mutagen_file["genre"][0]
            if genre not in self.genres:
                continue

            if self.file_transform is not None:
                song_file = self.file_transform(song_file)

            files.append(song_file)
            labels.append(genre_to_label[genre])

        data = self.files_to_data(files)

        return data, labels

    def create_dataset_from_playlist_genre(
        self, playlist_to_genre: Dict[str, str]
    ) -> (List[Path], List[int]):
        files_per_class = [[] for i in range(self.num_classes)]
        genre_to_label = {genre: i for i, genre in enumerate(self.genres)}

        # first we aggregate all files and genres
        for i, (playlist_id, genre) in enumerate(playlist_to_genre.items()):
            genre = playlist_to_genre[playlist_id]
            playlist = self.playlist_data[playlist_id]
            label = genre_to_label[genre]

            m3u_file = self.data_dir / Path(playlist["m3u_file"])
            with open(m3u_file, "r") as m3u:
                for song_path in m3u:
                    song_file = self.data_dir / Path(song_path)

                    if self.file_transform is not None:
                        song_file = self.file_transform(song_file)

                    if not song_file.exists():
                        continue

                    if song_file in files_per_class[label]:
                        continue

                    files_per_class[label].append(song_file)

        # then we can easier perform further processing on the files
        all_files, labels = self.flatten_file_array(files_per_class)

        print(f"Preprocessing complete")

        data = self.files_to_data(all_files)

        return data, labels

    def flatten_file_array(self, files_per_class):
        if self.even_classes:
            total_count = sum([len(class_files) for class_files in files_per_class])
            min_class_size = min([len(class_files) for class_files in files_per_class])

            print(
                f"Clamping dataset to {min_class_size} songs per class. "
                f"Removing {total_count - min_class_size * self.num_classes} songs."
            )
            self.class_size = min_class_size

            # shuffle the files so to randomly sample across all playlists for a given genre
            for files in files_per_class:
                shuffle(files)

            all_files = [
                file for files in files_per_class for file in files[:min_class_size]
            ]
            labels = [
                i
                for i, files in enumerate(files_per_class)
                for file in files[:min_class_size]
            ]

        else:
            all_files = [file for files in files_per_class for file in files]

            labels = [i for i, files in enumerate(files_per_class) for file in files]
        return all_files, labels

    def files_to_data(self, files):
        def file_to_data(f):
            try:
                d, sample_rate = torchaudio.load(f, num_frames=self.max_frames)
            except RuntimeError:
                return None

            if self.preprocess:
                d = self.preprocess(d)

            return d

        file_to_data(files[0]).shape
        data = [None] * len(files)

        for i, file in enumerate(tqdm(files, desc="Creating dataset", leave=True)):
            data[i] = file_to_data(file)
            self.ensure_enough_memory()

        stats = StatsRecorder()

        for d in data:
            stats.update(d)

        print(f"mean: {stats.mean}, std: {stats.std}")

        # normalizing the data across the full dataset imrpoved the models performance a lot
        for i, d in enumerate(data):
            data[i] = (d - stats.mean) / stats.std

        return data

    def create_dry_run_dataset(self) -> (List[Path], List[int]):
        data = []
        labels = []

        pbar = tqdm(desc="Creating dataset", total=10 * self.num_classes)
        for c in range(self.num_classes):
            for i in range(10):
                labels.append(c)

                # random 60s data
                d = torch.rand((1, 16000 * 60))
                if self.preprocess:
                    d = self.preprocess(d)
                data.append(d)

                pbar.update()
        pbar.close()

        self.class_size = 10

        return data, labels

    def aggregate_best_genres(self, num_classes) -> List[str]:
        """Select only the most common genres"""
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
            .head(num_classes)
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

        if free_memory < 1000:
            raise MemoryError(
                "Aborting. Less than 1GB available memory left on device. "
            )


class RepeatedLoader:
    """Increase the actual dataset size by running thorugh the full
    dataset multiple times."""

    def __init__(self, loader: DataLoader, repeat_count: int):
        self.loader = loader
        self.repeat_count = repeat_count

    def __iter__(self):
        for _ in range(self.repeat_count):
            yield from self.loader

    def __len__(self):
        return self.repeat_count * len(self.loader)


# taken and adapted from here
# https://colab.research.google.com/github/enzokro/clck10/blob/master/_notebooks/2020-09-10-Normalizing-spectrograms-for-deep-learning.ipynb
class StatsRecorder:
    def __init__(self, red_dims=(1, 2)):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims  # which mini-batch dimensions to average over
        self.nobservations = 0  # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std = data.std(dim=self.red_dims, keepdim=True)
            self.nobservations = 1
            self.ndimensions = data.shape[0]
        else:
            if data.shape[0] != self.ndimensions:
                raise ValueError("Data dims do not match previous observations.")

            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd = data.std(dim=self.red_dims, keepdim=True)

            # update number of observations
            m = self.nobservations * 1.0
            n = 1

            # update running statistics
            tmp = self.mean
            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = (
                m / (m + n) * self.std**2
                + n / (m + n) * newstd**2
                + m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            )
            self.std = torch.sqrt(self.std)

            # update total number of seen samples
            self.nobservations += n
