# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""01. SHA-256 of every data file the benchmarks read (manifests/data.tsv).

    python scripts/01_data_manifest.py --data-root /path/to/data            # compare with the manifest
    python scripts/01_data_manifest.py --data-root /path/to/data --write    # write it (the runs of the paper)

The files are the ones the loaders of this repository read (tabomics_paper.benchmarks), with the layout of
docs/benchmarks.md. Only file names, sizes and hashes are written, never their contents.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from tabomics_paper.benchmarks import hdlss, mlomics, shamir

from common import MANIFESTS, add_data_root, data_root, sha256_file, write_or_check


def data_files(root: Path) -> list[tuple[str, Path]]:
    """(suite, path relative to the data root) of every file the loaders read."""
    files = [("hdlss", Path(hdlss.FOLDER) / f"{name}.mat") for name in hdlss.DATASETS]
    for cancer in mlomics.CANCERS:
        folder = Path(mlomics.FOLDER) / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"
        files += [("mlomics", folder / f"{cancer}_{name}.csv") for name in mlomics.OMIC_FILES.values()]
        files.append(("mlomics", folder / f"{cancer}_label_num.csv"))
    for cancer in shamir.LABEL_COLUMNS:
        files += [("shamir", Path(shamir.FOLDER) / cancer / name) for name in shamir.OMIC_FILES.values()]
        files.append(("shamir", Path(shamir.FOLDER) / "clinical" / "clinical" / cancer))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--write", action="store_true", help="write the manifest instead of comparing with it")
    args = parser.parse_args()
    root = data_root(args)
    rows, missing = [], []
    for suite, path in data_files(root):
        full = root / path
        if not full.is_file():
            missing.append(str(path))
            continue
        rows.append({"suite": suite, "file": path.as_posix(), "bytes": full.stat().st_size,
                     "sha256": sha256_file(full)})  # fmt: skip
        print(f"{path.as_posix()}: {rows[-1]['sha256'][:12]}", flush=True)
    if missing:
        print(f"{len(missing)} files missing under {root}:\n" + "\n".join(f"  {m}" for m in missing))
        return 1
    agree = write_or_check(pd.DataFrame(rows), MANIFESTS / "data.tsv", ["file"], args.write)
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
