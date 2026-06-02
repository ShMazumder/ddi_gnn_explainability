"""
Download public DDI datasets (TWOSIDES, SIDER, STRING).
DrugBank XML must be obtained manually from https://go.drugbank.com (academic license).
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path

import yaml


def download_file(url, dest):
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # TWOSIDES (example URL from Tatonetti lab – check current link)
    twosides_url = "https://tatonettilab.org/resources/TWOSIDES/TWOSIDES.csv"
    twosides_dest = raw_dir / "TWOSIDES.csv"
    if not twosides_dest.exists():
        download_file(twosides_url, twosides_dest)

    # SIDER (MedDRA)
    sider_url = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
    sider_gz = raw_dir / "sider.tsv.gz"
    if not sider_gz.exists():
        download_file(sider_url, sider_gz)
        with gzip.open(sider_gz, 'rb') as f_in:
            with open(raw_dir / "sider.tsv", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # STRING protein-protein links (human)
    string_url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    string_gz = raw_dir / "9606.protein.links.v12.0.txt.gz"
    if not string_gz.exists():
        download_file(string_url, string_gz)
        with gzip.open(string_gz, 'rb') as f_in:
            with open(raw_dir / "9606.protein.links.v12.0.txt", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("Download complete. Please manually place DrugBank XML file as data/raw/drugbank.xml")


if __name__ == "__main__":
    main()
