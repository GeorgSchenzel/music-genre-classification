from pathlib import Path

import numpy as np
import seaborn as sn
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix

from mgclass import networks, analysis
from mgclass import MusicGenreDataset, RepeatedLoader
from mgclass.timer import Timer
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm, trange

from mgclass.utils import create_spectrogram, mp3_to_wav_location, sample_playlist_to_genre, create_crop
from textwrap import dedent


def some_experiment():
    dry_run = True
    data_shape = (128, 128)

    dataset = MusicGenreDataset(
        data_dir=Path("/home/georg/Music/ADL/"),
        preprocess=create_spectrogram(win_length=1024),
        transform=create_crop(data_shape),
        file_transform=mp3_to_wav_location,
        dry_run=dry_run,
        playlist_to_genre=sample_playlist_to_genre,
        max_frames=16000*60,
        even_classes=True
    )
    num_classes = dataset.num_classes

    model = networks.ResNet(num_classes)

    run = TrainingRun(dataset, model, epochs=10, dry_run=dry_run, repeat_count=10)
    run.summary()
    run.start()
    run.test()
    run.plot()


class TrainingRun:
    def __init__(
        self,
        dataset: MusicGenreDataset,
        model,
        epochs=100,
        batch_size=16,
        repeat_count=1,
        dry_run=False,
        optimizer=None
    ):
        self.dataset = dataset
        self.model = model
        self.epochs = epochs if not dry_run else 10
        self.batch_size = batch_size
        self.repeat_count = repeat_count if not dry_run else 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._prepare_data_loaders()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        self.already_ran = False

        # for plotting
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []

    def summary(self):
        print(f"Total Size: {len(self.dataset)}")
        print(f"Train Size: {len(self.train_dataset)}")
        print(f"Val Size: {len(self.test_dataset)}")
        print(f"Test Size: {len(self.test_dataset)}")
        print(f"Genres: {self.dataset.genres}")

    def start(self):
        if self.already_ran:
            raise Exception("Training already ran.")
        self.already_ran = True

        with Timer("Training"):
            self._train()

        print("")

    def _prepare_data_loaders(self):
        def create_indices(class_size, num_classes, per_class, offset):
            return [class_size * c + s + offset
                    for s in range(per_class)
                    for c in range(num_classes)]

        if self.dataset.class_size is None:
            raise Exception("Dataset should be evenly sized. Use even_classes=True")

        # in samples per class
        train_size = int(0.8 * self.dataset.class_size)
        val_size = (self.dataset.class_size - train_size) // 2
        test_size = self.dataset.class_size - train_size - val_size
        self.train_dataset = Subset(
            self.dataset,
            create_indices(self.dataset.class_size, self.dataset.num_classes, train_size, offset=0))

        self.val_dataset = Subset(
            self.dataset,
            create_indices(self.dataset.class_size, self.dataset.num_classes, val_size, offset=train_size))

        self.test_dataset = Subset(
            self.dataset,
            create_indices(self.dataset.class_size, self.dataset.num_classes, test_size, offset=train_size + val_size))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True
        )

        self.train_loader = RepeatedLoader(self.train_loader, self.repeat_count)
        self.val_loader = RepeatedLoader(self.val_loader, self.repeat_count)

    def _train(self):
        num_classes = self.dataset.num_classes
        train_acc = Accuracy("multiclass", num_classes=num_classes).to(self.device)
        val_acc = Accuracy("multiclass", num_classes=num_classes).to(self.device)

        pbar = tqdm(
            unit="epochs",
            total=self.epochs,
            leave=False,
            unit_scale=True
        )

        print(f"Starting training for {self.epochs} epochs")
        bar_step = 1/(len(self.train_loader) + len(self.val_loader))
        for epoch in range(self.epochs):
            pbar.set_description(desc=f"Epoch {epoch + 1:3d}/{self.epochs}")
            epoch_timer = Timer().start()

            batch_losses = []

            for data, label in self.train_loader:
                # Move to device
                data = data.to(self.device)
                label = label.to(self.device)

                self.model.train(True)
                self.model.zero_grad()

                # Forward pass
                outputs = self.model.forward(data)
                loss = self.loss_fn(outputs, label)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                self.model.train(False)

                batch_losses.append(loss.item())
                train_acc.update(outputs, label)

                pbar.update(bar_step)

            # validation

            val_losses = []
            for data, label in self.val_loader:
                # Move to device
                data = data.to(self.device)
                label = label.to(self.device)

                pred = self.model(data)

                val_loss = self.loss_fn(pred, label)
                val_losses.append(val_loss.item())
                val_acc.update(pred, label)

                pbar.update(bar_step)

            self.train_losses.append(np.array(batch_losses).mean())
            self.train_accuracies.append(float(train_acc.compute()))
            self.val_losses.append(np.array(val_losses).mean())
            self.val_accuracies.append(float(val_acc.compute()))

            # console output
            print(
                f"Epoch {epoch + 1:3d}/{self.epochs}, "
                f"train_loss: {self.train_losses[-1]:2.3f}, train_acc: {train_acc.compute():1.3f}, "
                f"val_loss: {self.val_losses[-1]:2.3f}, val_acc: {val_acc.compute():1.3f}, "
                f"in {epoch_timer.stop(False):2.2f}s"
            )

            train_acc.reset()
            val_acc.reset()

    def test(self):
        num_classes = self.dataset.num_classes
        test_acc = Accuracy("multiclass", num_classes=num_classes).to(self.device)

        y_pred = []
        y_true = []
        test_losses = []
        cm = ConfusionMatrix("multiclass", num_classes=self.dataset.num_classes).to(
            self.device
        )

        # remove the augmentation for test
        self.dataset.transform = None

        for data, label in self.test_loader:
            # Move to device
            data = data.to(self.device)
            label = label.to(self.device)

            count = 0
            pred = torch.zeros((1, num_classes)).to(self.device)
            for i in range(0, data.shape[3] - 128, 128):
                count += 1
                pred += self.model(data[:, :, :, i:i + 128])

            pred /= count

            test_loss = self.loss_fn(pred, label)
            test_losses.append(test_loss.item())
            test_acc.update(pred, label)

            cm.update(pred, label)

            y_true.extend(label.tolist())
            y_pred.extend(pred.tolist())

        self.confusion_matrix = cm.compute().cpu().numpy()
        self.test_acc = test_acc.compute()

        print(
            f"test_loss: {np.array(test_losses).mean():2.3f}, test_acc: {self.test_acc:3.3f}"
        )

    def plot(self, title="Training Run", additional_info=""):
        xx = np.arange(self.epochs)

        fig, ((ax_text, ax_cm), (ax_acc, ax_loss)) = plt.subplots(2, 2)
        fig.suptitle(title)

        ax_text.set_axis_off()
        text = dedent(f"""\
                Summary:
                Model: {type(self.model).__name__}
                Optimizer: {type(self.optimizer).__name__}
                
                Effective dataset size: {len(self.dataset)*self.repeat_count} samples
                Test accuracy: {self.test_acc:3.3f}
                
                {additional_info}
                """)
        ax_text.text(0.1, 0.9, text, verticalalignment="top")

        ax_cm.set_title("Confusion Matrix")
        df_cm = pd.DataFrame(
            self.confusion_matrix,
            index=[i for i in self.dataset.genres],
            columns=[i for i in self.dataset.genres],
        )
        sn.heatmap(df_cm, annot=True, ax=ax_cm, cbar=False, fmt='d')
        ax_cm.tick_params(rotation=45)
        ax_cm.set_xticklabels(ax_cm.get_xticklabels(), ha="right")
        ax_cm.set_yticklabels(ax_cm.get_yticklabels(), va="top")
        ax_cm.set(xlabel="predicted", ylabel="actual")

        ax_acc.set_title("Accuracy")
        ax_acc.set(xlabel="epoch", ylabel="accuracy")
        ax_acc.plot(xx, self.train_accuracies, label="train")
        ax_acc.plot(xx, self.val_accuracies, label="validate")
        ax_acc.set_ylim(bottom=0, top=1)
        ax_acc.legend()

        ax_loss.set_title("Loss")
        ax_loss.set(xlabel="epoch", ylabel="loss")
        ax_loss.plot(xx, self.train_losses, label="train")
        ax_loss.plot(xx, self.val_losses, label="validate")
        ax_loss.set_ylim(bottom=0)
        ax_loss.legend()

        fig.set_size_inches(10, 8)
        plt.tight_layout()
        plt.show()


async def main(args):
    some_experiment()
