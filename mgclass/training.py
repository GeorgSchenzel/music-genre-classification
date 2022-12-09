import sys
from pathlib import Path

import numpy as np
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torch.optim as optim
from torchmetrics import Accuracy

from mgclass.music_genre_dataset import MusicGenreDataset
from mgclass.timer import Timer


async def main(args):

    num_classes = 10
    sample_rate = 16000
    win_length = 2048
    hop_size = 512
    n_mels = 96
    crop_length_seconds = 15
    # Dataset

    """
    win_length defines the width of each chunk in terms of samples
    
    the duration of each chunk is win_length / sampling_rate
    the number of chunks is #total_frames / hop_length
    the overlap is win_length - hop_length / win_length 
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_size,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    random_crop = transforms.RandomCrop((n_mels, int(crop_length_seconds * sample_rate / hop_size)))

    def select_file(original: Path):
        new = original.parent / "wav_16k" / original.with_suffix(".wav").name

        return new

    with Timer("Dataset creation"):
        dataset = MusicGenreDataset(
            data_dir=Path("/home/georg/Music/ADL/"),
            preprocess=mel_spectrogram,
            transform=random_crop,
            file_transform=select_file,
            num_classes=num_classes,
        )

    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Total Size: {len(dataset)}")
    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(test_dataset)}")
    print(f"Test Size: {len(test_dataset)}")
    print(f"Genres: {dataset.genres}")

    # Model

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.conv1 = nn.Conv2d(1, 64, (64, 64), (2, 2), (3, 3), bias=False)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1),
    )

    # Move to Device (preferably GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}\n")

    model.to(device)

    # Setup

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=8, prefetch_factor=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=8
    )

    # Train

    acc_val = Accuracy("multiclass", num_classes=num_classes).to(device)
    acc_test = Accuracy("multiclass", num_classes=num_classes).to(device)

    epochs = 100
    print(f"Starting training for {epochs} epochs\n")
    sys.stdout.flush()
    with Timer("Training"):
        for epoch in range(epochs):
            epoch_timer = Timer().start()

            batch_losses = []

            for data, label in train_loader:
                # Move to device
                data = data.to(device)
                label = label.to(device)

                model.train(True)
                model.zero_grad()

                # Forward pass
                outputs = model.forward(data)
                loss = loss_fn(outputs, label)

                # Backward pass
                loss.backward()
                optimizer.step()

                model.train(False)
                batch_losses.append(loss.item())

            batch_losses = np.array(batch_losses)
            mean = round(np.mean(batch_losses), 3)
            std = round(np.std(batch_losses), 4)

            # validation

            for data, label in val_loader:
                # Move to device
                data = data.to(device)
                label = label.to(device)

                pred = model(data)
                acc_val.update(pred, label)

            # console output
            print(
                f"Epoch {epoch:3d}/{epochs}, Loss: {mean:2.3f} +- {std:1.4f}, Accuracy: {acc_val.compute():3.3f}, in {epoch_timer.stop(False):2.2f}s"
            )

    for data, label in test_loader:
        # Move to device
        data = data.to(device)
        label = label.to(device)

        pred = model(data)
        acc_test.update(pred, label)

    print(f"Test Accuracy: {acc_test.compute()}")
