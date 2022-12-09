from pathlib import Path
import yaml
from spotdj.spotdj import Spotdj


class DataDownloader:
    def __init__(self, path: Path, playlist_file_location: Path):
        self.path = path
        self.playlist_urls = self.read_playlists_file(playlist_file_location)

    async def download_all(self):
        with Spotdj(Path.home() / "Music" / "ADL", use_rym_metadata=False) as sdj:
            for playlist in self.playlist_urls:
                await sdj.download_playlist(playlist)

    @staticmethod
    def read_playlists_file(location: Path):
        with open(location, "r") as f:
            data = yaml.safe_load(f)

            return data


async def main(args):
    await DataDownloader(Path(args.path), Path(args.playlists_file)).download_all()
