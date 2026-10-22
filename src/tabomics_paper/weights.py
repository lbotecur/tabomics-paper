# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""The weights of the other models, each fixed to a source, a revision and a SHA-256, downloaded once and checked.

The packages of these models download their weights by file name, without fixing a revision: a new file under the
same name would change the results. Here every file is fetched at a fixed commit (Hugging Face) or release (GitHub)
and checked against its SHA-256 before a model gets it, through its model_path. The weights of TabOmics are fixed
in the same way by tabomics itself (tabomics.hub).
"""

from __future__ import annotations

import functools
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class Weights:
    """A file of weights: its name, where it comes from, at which revision, and its SHA-256."""

    file: str
    source: str  # a Hugging Face repository, or the URL of a GitHub release asset
    revision: str  # a commit of the Hugging Face repository, or the release tag
    sha256: str


WEIGHTS = {
    "tabpfn-3.5": Weights(
        "tabpfn-v3.5-20260909.safetensors",
        "Prior-Labs/tabpfn_3_5",
        "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    ),
    "tabpfn-v2-gn2p4bpt": Weights(
        "tabpfn-v2-classifier-gn2p4bpt.ckpt",
        "Prior-Labs/TabPFN-v2-clf",
        "f851f2a3c941544733b712d8c0f96dfae9b28862",
        "de01a095da7a083b25a74d1f7d85de524e8bd6137936092feab9790673ad9215",
    ),
    "tabicl-v2": Weights(
        "tabicl-classifier-v2-20260212.ckpt",
        "jingang/TabICL",
        "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    ),
    "tabpfn-wide-5k": Weights(
        "tabpfn-wide-v2-5k.pt",
        "https://github.com/not-a-feature/TabPFN-Wide/releases/download/v0.4.1/tabpfn-wide-v2-5k.pt",
        "v0.4.1",
        "b6f67c4abb9d7bb1fc1a57514848401bbe6326b7aae7fd050621687eac7be01b",
    ),
}
# Where the weights are kept, as <model>/<revision>/<file>. Those from Hugging Face are real files there, not links
# into its cache: TabPFN resolves links, and a link's target there has no extension, so it would not load a
# .safetensors file as one.
CACHE = Path.home() / ".cache" / "tabomics-paper" / "weights"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def weights_path(model: str) -> Path:
    """Local path of the weights of a model, downloaded at their fixed revision and checked against their SHA-256."""
    weights = WEIGHTS[model]
    path = CACHE / model / weights.revision / weights.file
    if weights.source.startswith("https://") and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(weights.source, partial)
        partial.rename(path)
    elif not path.exists():
        hf_hub_download(weights.source, weights.file, revision=weights.revision, local_dir=path.parent)
    found = sha256_of(path)
    if found != weights.sha256:
        raise RuntimeError(f"{path}: SHA-256 {found}, but the weights of {model} are {weights.sha256}")
    return path
