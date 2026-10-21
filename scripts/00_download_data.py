# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""00. Downloads the data of the three benchmarks into the data folder, in the layout the loaders read.

    python scripts/00_download_data.py --data-root /path/to/data

- HDLSS: the 15 .mat files of scikit-feature (https://jundongl.github.io/scikit-feature/datasets.html)
  -> <data_root>/HDLSS/<name>.mat
- MLOmics: the classification datasets ("Original") of the Hugging Face dataset AIBIC/MLOmics, at a fixed revision
  -> <data_root>/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/GS-<cancer>/Original/
- Shamir: breast, gbm and sarcoma, and the clinical data, of the multi-omics cancer benchmark of Rappoport and
  Shamir (https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html)
  -> <data_root>/Shamir/<cancer>/{exp,methy,mirna} and <data_root>/Shamir/clinical/clinical/<cancer>

Files already present are kept. Needs internet access. The data are not redistributed by this repository; each source
has its own terms. Afterwards, scripts/01_data_manifest.py checks that the files are those used in the paper.
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tabomics_paper.benchmarks import hdlss, mlomics, shamir

from common import add_data_root, data_root

HDLSS_URL = "https://jundongl.github.io/scikit-feature/files/datasets/{name}.mat"
SHAMIR_URL = "https://acgt.cs.tau.ac.il/multi_omic_benchmark/data/{name}.zip"
MLOMICS_REPOSITORY = "AIBIC/MLOmics"
MLOMICS_REVISION = None  # commit of the dataset used in the paper; None takes the latest and prints its commit


def download(url: str, path: Path) -> None:
    """Downloads url to path, through a temporary file so that an interrupted download leaves nothing behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)


def find(folder: Path, name: str) -> Path:
    """The single file called name under folder (macOS metadata folders ignored)."""
    found = [p for p in folder.rglob(name) if p.is_file() and "__MACOSX" not in p.parts]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one file {name} in the archive, found {len(found)}")
    return found[0]


def download_hdlss(root: Path) -> None:
    for name in hdlss.DATASETS:
        path = root / hdlss.FOLDER / f"{name}.mat"
        if not path.exists():
            download(HDLSS_URL.format(name=name), path)
        print(f"HDLSS/{name}.mat", flush=True)


def download_shamir(root: Path) -> None:
    folder = root / shamir.FOLDER
    with tempfile.TemporaryDirectory() as tmp:
        for cancer in shamir.LABEL_COLUMNS:
            targets = [folder / cancer / name for name in shamir.OMIC_FILES.values()]
            if not all(t.exists() for t in targets):
                archive = Path(tmp) / f"{cancer}.zip"
                download(SHAMIR_URL.format(name=cancer), archive)
                zipfile.ZipFile(archive).extractall(Path(tmp) / cancer)
                for target in targets:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(find(Path(tmp) / cancer, target.name), target)
            print(f"Shamir/{cancer}", flush=True)
        targets = [folder / "clinical" / "clinical" / cancer for cancer in shamir.LABEL_COLUMNS]
        if not all(t.exists() for t in targets):
            archive = Path(tmp) / "clinical.zip"
            download(SHAMIR_URL.format(name="clinical"), archive)
            zipfile.ZipFile(archive).extractall(Path(tmp) / "clinical")
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(find(Path(tmp) / "clinical", target.name), target)
        print("Shamir/clinical", flush=True)


def download_mlomics(root: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    revision = MLOMICS_REVISION or HfApi().dataset_info(MLOMICS_REPOSITORY).sha
    snapshot_download(
        repo_id=MLOMICS_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"Main_Dataset/Classification_datasets/GS-{cancer}/Original/*" for cancer in mlomics.CANCERS],
        local_dir=root / mlomics.FOLDER,
    )
    print(f"MLOmics: {MLOMICS_REPOSITORY} at revision {revision}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--suites", nargs="+", default=["hdlss", "mlomics", "shamir"])
    args = parser.parse_args()
    if args.data_root:
        Path(args.data_root).expanduser().mkdir(parents=True, exist_ok=True)
    root = data_root(args)
    steps = {"hdlss": download_hdlss, "mlomics": download_mlomics, "shamir": download_shamir}
    for suite in args.suites:
        steps[suite](root)
    print("Next: python scripts/01_data_manifest.py, to check the files against those of the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
