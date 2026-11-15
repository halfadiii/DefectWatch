"""Download and unpack MVTec AD into datasets/MVTecAD.

The link baked into anomalib 2.7 is dead, so this uses the one on MVTec's own
downloads page (https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
and checks the file against the SHA-256 that anomalib ships.

MVTec AD is licensed CC BY-NC-SA 4.0. Credit: Bergmann et al., CVPR 2019.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = (
    "https://www.mydrive.ch/shares/163178/c8212c2166a9e4dfe22dd1bed476a345/"
    "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
)
SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
ROOT = Path(__file__).resolve().parents[1] / "datasets" / "MVTecAD"
ARCHIVE = ROOT.parent / "mvtec_anomaly_detection.tar.xz"
CHUNK = 4 * 1024 * 1024


def download() -> None:
    """Stream the archive to disk, resuming a partial file if one is there."""
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    have = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    request = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(request, timeout=60)  # noqa: S310
    except urllib.error.HTTPError as err:
        if err.code == 416:  # asked for bytes past the end: the file is already complete
            return
        raise
    resumed = response.status == 206
    total = int(response.headers["Content-Length"]) + (have if resumed else 0)
    done = have if resumed else 0
    next_report = done + 250 * 1024 * 1024
    with response, ARCHIVE.open("ab" if resumed else "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"  {done / 1e9:.2f} of {total / 1e9:.2f} GB", flush=True)
                next_report += 250 * 1024 * 1024


def verify() -> None:
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        sys.exit(f"Checksum mismatch: got {digest.hexdigest()}. Delete {ARCHIVE} and run again.")


def main() -> None:
    if (ROOT / "bottle" / "train" / "good").is_dir() and (ROOT / "zipper" / "test").is_dir():
        print(f"MVTec AD is already at {ROOT}")
        return
    print("Downloading (5.3 GB)...", flush=True)
    download()
    print("Checking the file...", flush=True)
    verify()
    print("Unpacking...", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        tar.extractall(ROOT, filter="data")
    ARCHIVE.unlink()
    print(f"Done: {ROOT}")


if __name__ == "__main__":
    main()
