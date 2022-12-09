from pathlib import Path

import torch
import torchaudio.transforms as T
import torchaudio
import torchvision.transforms
from matplotlib import pyplot as plt
import librosa


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)

    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def spectrogram_transform(waveform, sample_rate):
    n_fft = 1024
    win_length = None
    hop_length = 512

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    spec = spectrogram(waveform)
    plot_spectrogram(
        spec[0],
        title="torchaudio",
    )


def mel_spectrogram_transform(waveform, sample_rate):
    n_fft = 1024
    win_length = 512
    hop_length = 512
    n_mels = 96

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)
    plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")


async def main(args):
    frame_offset, num_frames = 10 * 44100, 15 * 44100

    waveform, sample_rate = torchaudio.load(
        Path("/home/georg/Music/ADL/Disco Fries - Family Affair.mp3"),
    )

    print_stats(waveform, sample_rate)
    t = torchvision.transforms.RandomCrop((1, 15 * sample_rate))
    waveform = t(waveform)

    print_stats(waveform, sample_rate)

    # spectrogram_transform(waveform, sample_rate)
    mel_spectrogram_transform(waveform, sample_rate)
