from pathlib import Path


def mp3_to_wav_location(mp3: Path, subfolder="wav_16k") -> Path:
    return mp3.parent / subfolder / mp3.with_suffix(".wav").name
