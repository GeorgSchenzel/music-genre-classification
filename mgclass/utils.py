from pathlib import Path
import torchaudio.transforms as a_transforms
import torchvision.transforms as v_transforms

sample_playlist_to_genre = {
    "18vUeZ9BdtMRNV6gI8RnR6": "Techno",
    "0rvPxkmlJZ5EX4JXutyP6I": "Tech House",
    "37i9dQZF1DWTU3Zl0elDUa": "House Classic",
    "6vDGVr652ztNWKZuHvsFvx": "Deep House",
    "068WHS0zOWsqvn2uIBYb5D": "DnB",
    "0ZOspi3XrshQrmVzmlQFx6": "Liquid DnB",
    "4aKW9X7zIju4ijCSL7MR7T": "Future Rave",
    "6yyFZPfg3pU3d3IHNpaNKI": "Tech House",
    "37i9dQZF1DWUbycBFSWTh7": "Deep House",
    "1BV6g3eusEscbN3qaTA681": "House Classic",
    "27vbShxLFuNJ3d2Z0u67LT": "Future Rave",
    "0R29couUMdcX6JUDGFFapa": "Psytrance",
    "7D25GYqznqpZJP1GCjMtF1": "Psytrance",
    "3ZG1CAQ811cCiSce3J0uz5": "Psytrance",
    "502B3oSddAW7czuMMvxITT": "Future Rave",
    "5NlU4dHwuK9JrhpuwYGllf": "Liquid DnB",
    "7mwPa6HjqoiUrsk3C2Hitk": "Techno",
}


def mp3_to_wav_location(mp3: Path, subfolder="wav_16k") -> Path:
    return mp3.parent / subfolder / mp3.with_suffix(".wav").name


def create_spectrogram(sample_rate=16000, n_mels=128, win_length=1024, overlap=0.75):
    """
    win_length defines the width of each chunk in terms of samples

    the duration of each chunk is win_length / sampling_rate
    the number of chunks is #total_frames / hop_length
    the overlap is win_length - hop_length / win_length

    thus we can calculate the receptive field of the first layer by:
        seconds = win + n * hop / sample_rate
                = win * (1 + n * (1 - overlap) / sample_rate

    a single bin covers win_length / sample_rate seconds of out data

    a win_length of 1024 @ 16khz covers 64ms
    """

    hop_size = int(win_length - overlap * win_length)

    return a_transforms.MelSpectrogram(
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


def create_crop(shape):
    return v_transforms.RandomCrop(shape)
