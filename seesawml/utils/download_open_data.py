import argparse
import json
import logging
import subprocess
from pathlib import Path

import atlasopenmagic as atom

from seesawml.utils.loggers import setup_logger


def download_open_data(download_dir: Path, release: str, skim: str) -> None:
    atom.set_release(release)

    for dataset in atom.available_datasets():
        logging.info(f"Downloading dataset {dataset}")

        # Save the dataset metadata
        with (download_dir / f"{dataset}.metadata.json").open("w") as f:
            json.dump(atom.get_metadata(dataset), f, indent=4)

        # Download the files for the dataset
        for file in atom.get_urls(dataset, skim=skim, protocol="root"):
            logging.info(file)
            subprocess.run(["xrdcp", file, download_dir], check=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="download_open_data", description="Download ATLAS Open Data")

    parser.add_argument(
        "-d",
        "--download-dir",
        type=Path,
        required=True,
        help="directory to download the files to",
    )
    parser.add_argument(
        "-r",
        "--release",
        type=str,
        default="2025e-13tev-beta",
        help="release of the ATLAS Open Data to download. Default: 2025e-13tev-beta",
    )
    parser.add_argument(
        "-s",
        "--skim",
        type=str,
        default="1LMET30",
        help="skim of the release to download. Default: 1LMET30",
    )

    return parser.parse_args()


def main() -> None:
    setup_logger()

    args: argparse.Namespace = parse_arguments()

    if not args.download_dir.exists():
        logging.info(f"Creating download directory {args.download_dir}")
        args.download_dir.mkdir(parents=True)

    download_open_data(args.download_dir, args.release, args.skim)


if __name__ == "__main__":
    main()
