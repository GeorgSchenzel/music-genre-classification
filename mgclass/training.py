import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torch.optim as optim
from torchmetrics import Accuracy

from mgclass.MyNet import MyNet
from mgclass.ResNet import ResNet
from mgclass.music_genre_dataset import MusicGenreDataset
from mgclass.timer import Timer
from torchinfo import summary
from matplotlib import pyplot as plt
from tqdm import tqdm


def repeat(loader, times):
    for _ in range(times):
        for x in loader:
            yield x


async def main(args):
    batch_size = 16
    data_shape = (128, 256)
    repeat_count = 5

    num_classes = 10
    sample_rate = 16000
    win_length = 2048
    hop_size = 512
    n_mels = data_shape[0]

    # Model

    model = ResNet(num_classes)
    #model = MyNet(num_classes)
    summary(model, input_size=(batch_size, 1) + data_shape)

    # Move to Device (preferably GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}\n")

    model.to(device)

    # Dataset

    """
    win_length defines the width of each chunk in terms of samples
    
    the duration of each chunk is win_length / sampling_rate
    the number of chunks is #total_frames / hop_length
    the overlap is win_length - hop_length / win_length
    
    thus we can calculate the receptive field of the first layer by:
        seconds = kernel_size * hop_size / sample_rate
    
    or a single bin covers hop_size / sample_rate seconds of out data
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

    random_crop = transforms.RandomCrop((n_mels, data_shape[1]))

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

    # Setup

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # Train

    acc_train = Accuracy("multiclass", num_classes=num_classes).to(device)
    acc_val = Accuracy("multiclass", num_classes=num_classes).to(device)
    acc_test = Accuracy("multiclass", num_classes=num_classes).to(device)

    # for plotting later
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    epochs = 100
    print(f"Starting training for {epochs} epochs\n")
    sys.stdout.flush()
    with Timer("Training"):
        for epoch in range(epochs):
            epoch_timer = Timer().start()

            batch_losses = []

            pbar = tqdm(desc=f"Epoch {epoch:3d}/{epochs}", unit="batches", total=len(train_loader) * repeat_count, leave=False)
            for data, label in repeat(train_loader, repeat_count):
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

                acc_train.update(outputs, label)
                batch_losses.append(loss.item())

                pbar.update()

            # validation

            for data, label in repeat(val_loader, repeat_count):
                # Move to device
                data = data.to(device)
                label = label.to(device)

                pred = model(data)
                loss_val = loss_fn(pred, label)
                acc_val.update(pred, label)

            train_accuracies.append(float(acc_train.compute()))
            val_accuracies.append(float(acc_val.compute()))
            train_losses.append(np.array(batch_losses).mean())
            val_losses.append(float(loss_val))

            # console output
            pbar.close()
            print(
                f"Epoch {epoch:3d}/{epochs}, "
                f"train_loss: {train_losses[-1]:2.3f}, train_acc: {acc_train.compute():3.3f}, "
                f"val_loss: {loss_val:2.3f}, val_acc: {acc_val.compute():3.3f}, "
                f"in {epoch_timer.stop(False):2.2f}s"
            )

            acc_train.reset()
            acc_val.reset()

    for data, label in repeat(test_loader, 10):
        # Move to device
        data = data.to(device)
        label = label.to(device)

        pred = model(data)
        acc_test.update(pred, label)

    print(f"Test Accuracy: {acc_test.compute()}")

    xx = np.arange(epochs)
    plt.title("Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(xx, train_accuracies, label="train")
    plt.plot(xx, val_accuracies, label="validate")
    plt.legend()
    plt.show()

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(xx, train_losses, label="train")
    plt.plot(xx, val_losses, label="validate")
    plt.legend()
    plt.show()
