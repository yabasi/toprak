# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Deney seed'i, ortam kaydı ve veri parmak izi yardımcıları."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import torch


MANIFEST_VERSION = "toprak-experiment-v1"
DATA_EXTENSIONS = {".json", ".jsonl", ".txt", ".bin", ".npy", ".npz"}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Python, NumPy ve PyTorch RNG'lerini tek seed ile başlat."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _candidate_manifests(data_path: str) -> list:
    data_path = os.path.abspath(data_path)
    candidates = []
    if os.path.isdir(data_path):
        candidates.extend([
            os.path.join(data_path, "manifest.json"),
            os.path.join(data_path, "corpus_manifest.json"),
            os.path.join(os.path.dirname(data_path), "corpus_manifest.json"),
        ])
    return [path for path in candidates if os.path.isfile(path)]


def _data_files(data_path: str) -> list:
    if os.path.isfile(data_path):
        return [os.path.abspath(data_path)]
    files = []
    for root, dirs, names in os.walk(data_path):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            path = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() in DATA_EXTENSIONS:
                files.append(os.path.abspath(path))
    return files


def fingerprint_data(data_path: str, mode: str = "auto") -> dict:
    """Manifest veya tüm içerik üzerinden sıralı veri parmak izi üret."""
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Veri parmak izi yolu bulunamadı: {data_path}")
    if mode not in {"auto", "manifest", "full", "metadata", "off"}:
        raise ValueError(f"Desteklenmeyen veri parmak izi modu: {mode}")
    if mode == "off":
        return {"mode": "off", "path": data_path, "sha256": None, "files": 0}

    manifests = _candidate_manifests(data_path)
    effective_mode = mode
    if mode == "auto":
        effective_mode = "manifest" if manifests else "full"
    if effective_mode == "manifest":
        if not manifests:
            raise FileNotFoundError(
                "Manifest parmak izi istendi fakat manifest.json/corpus_manifest.json bulunamadı"
            )
        files = manifests
    else:
        files = _data_files(data_path)
    if not files:
        raise ValueError(f"Parmak izi alınacak veri dosyası bulunamadı: {data_path}")

    digest = hashlib.sha256()
    entries = []
    for path in sorted(files):
        relative = os.path.relpath(path, data_path if os.path.isdir(data_path) else os.path.dirname(data_path))
        size = os.path.getsize(path)
        if effective_mode == "metadata":
            file_digest = None
            payload = f"{relative}\0{size}\n".encode("utf-8")
        else:
            file_digest = file_sha256(path)
            payload = f"{relative}\0{size}\0{file_digest}\n".encode("utf-8")
        digest.update(payload)
        entries.append({"path": relative, "size": size, "sha256": file_digest})
    return {
        "mode": effective_mode,
        "path": data_path,
        "sha256": digest.hexdigest(),
        "files": len(entries),
        "entries": entries,
    }


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(project_root: str) -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=project_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def runtime_metadata(project_root: str) -> dict:
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
    installed_packages = {
        distribution.metadata.get("Name", "unknown"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sentencepiece": _package_version("sentencepiece"),
        },
        "installed_packages": dict(sorted(installed_packages.items(), key=lambda item: item[0].lower())),
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_device": cuda_device,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git": _git_metadata(project_root),
    }


def build_experiment_manifest(
    project_root: str,
    training_recipe: dict,
    tokenizer_path: str,
    data_path: str,
    data_fingerprint_mode: str = "auto",
    argv: Iterable[str] = (),
) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "argv": list(argv),
        "training_recipe": training_recipe,
        "tokenizer": {
            "path": os.path.abspath(tokenizer_path),
            "sha256": file_sha256(tokenizer_path),
        },
        "data": fingerprint_data(data_path, data_fingerprint_mode),
        "runtime": runtime_metadata(project_root),
    }


def write_manifest(manifest: dict, *directories: str) -> list:
    paths = []
    for directory in dict.fromkeys(os.path.abspath(value) for value in directories):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "experiment_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        paths.append(path)
    return paths
