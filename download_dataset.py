#!/usr/bin/env python3
"""
Download script for MURA dataset
Run this after cloning to get the dataset
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# MURA dataset URL (you'll need to find the actual download link)
MURA_URL = "https://example.com/mura-v1.1.zip"  # Replace with actual URL
DATA_DIR = Path("data")

def download_and_extract(url, extract_to):
    """Download and extract zip file"""
    zip_path = extract_to / "mura_dataset.zip"

    print(f"Downloading MURA dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Clean up
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    download_and_extract(MURA_URL, DATA_DIR)