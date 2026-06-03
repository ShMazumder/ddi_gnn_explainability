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

    # TWOSIDES (working S3 download link)
    twosides_url = "https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/TWOSIDES.csv.gz"
    twosides_gz = raw_dir / "TWOSIDES.csv.gz"
    twosides_dest = raw_dir / "TWOSIDES.csv"
    if not twosides_dest.exists():
        if not twosides_gz.exists():
            download_file(twosides_url, twosides_gz)
        print(f"Extracting {twosides_gz} -> {twosides_dest}")
        with gzip.open(twosides_gz, 'rb') as f_in:
            with open(twosides_dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # DrugBank XML (working HuggingFace link)
    drugbank_url = "https://huggingface.co/datasets/agenticx/DrugBank/resolve/main/drugbank_full_database.xml"
    drugbank_dest = raw_dir / "drugbank.xml"
    if not drugbank_dest.exists():
        download_file(drugbank_url, drugbank_dest)

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

    # STRING protein aliases (human)
    aliases_url = "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
    aliases_gz = raw_dir / "9606.protein.aliases.v12.0.txt.gz"
    aliases_dest = raw_dir / "protein.aliases.v12.0.txt"
    if not aliases_dest.exists():
        if not aliases_gz.exists():
            download_file(aliases_url, aliases_gz)
        print(f"Extracting {aliases_gz} -> {aliases_dest}")
        with gzip.open(aliases_gz, 'rb') as f_in:
            with open(aliases_dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("Download complete. All datasets (TWOSIDES, SIDER, STRING, and DrugBank XML) have been downloaded successfully.")


if __name__ == "__main__":
    main()
