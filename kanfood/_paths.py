"""Repo-relative path resolution.

Keeps the package portable and reproducible: no hardcoded absolute paths. Every
data/output location resolves to a path *relative to the repository root*, and can
be overridden with the matching environment variable (useful when the data or
results live elsewhere, e.g. the private tahini spectra).
"""
import os
from pathlib import Path

# kanfood/ sits directly under the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]


def data_path(env_var: str, *relative: str) -> Path:
    """Return ``$env_var`` if it is set, otherwise ``REPO_ROOT / relative...``."""
    override = os.environ.get(env_var)
    return Path(override) if override else REPO_ROOT.joinpath(*relative)


def first_existing(env_var: str, *candidates: str) -> Path:
    """Return ``$env_var`` if set; else the first ``REPO_ROOT/candidate`` that exists;
    else ``REPO_ROOT/<first candidate>`` (the canonical default location).

    Lets a file resolve in either the public-release layout (e.g. ``data/tahini/...``)
    or the working-copy layout (repo root) without an environment variable.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    for rel in candidates:
        p = REPO_ROOT.joinpath(rel)
        if p.exists():
            return p
    return REPO_ROOT.joinpath(candidates[0])
