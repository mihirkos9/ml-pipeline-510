"""Data loading with a local cache and a SHA-256 integrity check.

Dataquest NBA MVP dataset (1991-2021), scraped from Basketball-Reference.
Pinned to a fixed git commit AND verified by hash, so everyone gets byte-identical data.

First run downloads the CSV into data/ (gitignored); later runs read the cached copy.
"""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DATA_REPO = "dataquestio/project-walkthroughs"
DATA_COMMIT = "571e6565edb7330c65a1dcb35033dac5856ffd6d"
DATA_URL = (f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_COMMIT}"
            "/mvp/player_mvp_stats.csv")
DATA_SHA256 = "5e1eb96717211d7e988fc2e68f3261608e7e4c7506bea197d3bd449cdb0ab3f0"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "player_mvp_stats.csv"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cache_is_valid() -> bool:
    return DATA_PATH.exists() and _sha256(DATA_PATH.read_bytes()) == DATA_SHA256


def download_data() -> None:
    """Download the pinned CSV, verify its hash, and only then save it to data/."""
    print(f"Downloading dataset (pinned commit {DATA_COMMIT[:7]}) ...")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        content = response.read()
    actual = _sha256(content)
    if actual != DATA_SHA256:
        raise RuntimeError(
            "Checksum mismatch: the downloaded data is not the expected file.\n"
            f"  expected {DATA_SHA256}\n  got      {actual}\n"
            "Nothing was saved. Do not use unverified data.")
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(content)


def load_data() -> pd.DataFrame:
    """Return the raw player-season table (one row per player per season)."""
    if not _cache_is_valid():      # missing, or edited/corrupted -> fetch a fresh verified copy
        download_data()
    return pd.read_csv(DATA_PATH)
