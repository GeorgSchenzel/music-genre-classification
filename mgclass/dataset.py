import json
from pathlib import Path
from random import shuffle
from typing import List, Dict

import numpy as np
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
        playlist_to_genre: Dict[str, str] = None, # should be playlistid: genre label,
        max_frames=-1,
        even_classes=True
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

        with Timer("Dataset creation"):

            if dry_run:
                self.genres = sorted(self.aggregate_best_genres(num_classes))
                self.num_classes = num_classes
                self.data, self.labels = self.create_dry_run_dataset()

            # get genres from this provided dict
            elif playlist_to_genre is not None:
                print(f"Using genre from playlist source")
                self.genres = sorted(set(playlist_to_genre.values()))
                self.num_classes = len(self.genres)
                self.data, self.labels = self.create_dataset_from_playlist_genre(playlist_to_genre)

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

    def create_dataset_from_playlist_genre(self, playlist_to_genre: Dict[str, str]) -> (List[Path], List[int]):
        files_per_class = [[] for i in range(self.num_classes)]
        genre_to_label = {genre: i for i, genre in enumerate(self.genres)}

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

        # show duplicates
        for i, files1 in enumerate(files_per_class):
            for j, files2 in enumerate(files_per_class):
                if i >= j:
                    continue

                duplicates = len([f for f in files1 if f in files2])
                if duplicates > 0:
                    print(f"Dups for {i}-{j}: {duplicates:3d}")

        all_files, labels = self.flatten_file_array(files_per_class)

        print(f"Preprocessing complete")

        data = self.files_to_data(all_files)

        return data, labels

    def flatten_file_array(self, files_per_class):
        if self.even_classes:
            total_count = sum([len(class_files) for class_files in files_per_class])
            min_class_size = min([len(class_files) for class_files in files_per_class])

            print(f"Clamping dataset to {min_class_size} songs per class. "
                  f"Removing {total_count - min_class_size * self.num_classes} songs.")

            # shuffle the files so to randomly sample across all playlists for a given genre
            for files in files_per_class:
                shuffle(files)

            all_files = [file
                         for files in files_per_class
                         for file in files[:min_class_size]]
            labels = [i
                      for i, files in enumerate(files_per_class)
                      for file in files[:min_class_size]]

        else:
            all_files = [file
                         for files in files_per_class
                         for file in files]

            labels = [i
                      for i, files in enumerate(files_per_class)
                      for file in files]
        return all_files, labels

    def files_to_data(self, files):
        data = [None] * len(files)
        for i, file in enumerate(tqdm(files, desc="Creating dataset")):
            try:
                d, sample_rate = torchaudio.load(file, num_frames=self.max_frames)
            except RuntimeError:
                continue

            if self.preprocess:
                d = self.preprocess(d)

            data[i] = d

            self.ensure_enough_memory()

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

        return data, labels

    def aggregate_best_genres(self, num_classes) -> List[str]:
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
    def __init__(self, loader: DataLoader, repeat_count: int):
        self.loader = loader
        self.repeat_count = repeat_count

    def __iter__(self):
        for _ in range(self.repeat_count):
            yield from self.loader

    def __len__(self):
        return self.repeat_count * len(self.loader)
