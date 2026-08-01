# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Kaynak grupları ve lineer curriculum mixture sampler."""

from __future__ import annotations

import json
import math
import os
import re
from bisect import bisect_right
from typing import Dict, Optional

import torch
from torch.utils.data import Sampler


MIXTURE_VERSION = "toprak-mixture-v1"


def _validate_weight(value, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} sonlu ve negatif olmayan sayı olmalı")
    return value


def validate_mixture_config(config: dict) -> dict:
    if config.get("version", MIXTURE_VERSION) != MIXTURE_VERSION:
        raise ValueError(f"Desteklenmeyen mixture version: {config.get('version')}")
    groups = config.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Mixture config en az bir groups girdisi içermeli")
    normalized = {}
    default_groups = 0
    claimed_sources = set()
    for name, group in groups.items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"Geçersiz mixture grup adı: {name!r}")
        if not isinstance(group, dict):
            raise ValueError(f"Mixture grubu nesne olmalı: {name}")
        sources = [str(source) for source in group.get("sources", [])]
        duplicates = claimed_sources.intersection(sources)
        if duplicates:
            raise ValueError(f"Kaynaklar birden fazla grupta: {sorted(duplicates)}")
        claimed_sources.update(sources)
        is_default = bool(group.get("default", False))
        default_groups += int(is_default)
        initial_weight = _validate_weight(
            group.get("initial_weight", 1.0), f"{name}.initial_weight"
        )
        final_weight = _validate_weight(
            group.get("final_weight", 1.0), f"{name}.final_weight"
        )
        if initial_weight == 0 and final_weight == 0:
            raise ValueError(f"{name} grubu tüm schedule boyunca sıfır ağırlıklı")
        normalized[name] = {
            "sources": sources,
            "default": is_default,
            "initial_weight": initial_weight,
            "final_weight": final_weight,
        }
    if default_groups != 1:
        raise ValueError("Mixture config tam olarak bir default grup içermeli")
    if sum(group["initial_weight"] for group in normalized.values()) <= 0:
        raise ValueError("Mixture initial ağırlık toplamı pozitif olmalı")
    if sum(group["final_weight"] for group in normalized.values()) <= 0:
        raise ValueError("Mixture final ağırlık toplamı pozitif olmalı")
    curriculum_steps = int(config.get("curriculum_steps", 0))
    if curriculum_steps < 0:
        raise ValueError("curriculum_steps negatif olamaz")
    return {
        "version": MIXTURE_VERSION,
        "curriculum_steps": curriculum_steps,
        "groups": normalized,
    }


def load_mixture_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return validate_mixture_config(json.load(handle))


def legacy_curriculum_config(
    high_quality_sources,
    curriculum_steps: int,
    initial_high_weight: float = 0.8,
    final_high_weight: float = 0.4,
) -> dict:
    initial_high_weight = _validate_weight(initial_high_weight, "initial_high_weight")
    final_high_weight = _validate_weight(final_high_weight, "final_high_weight")
    if initial_high_weight > 1 or final_high_weight > 1:
        raise ValueError("Legacy curriculum HQ ağırlıkları 0 ile 1 arasında olmalı")
    return validate_mixture_config({
        "version": MIXTURE_VERSION,
        "curriculum_steps": curriculum_steps,
        "groups": {
            "high_quality": {
                "sources": list(high_quality_sources),
                "initial_weight": initial_high_weight,
                "final_weight": final_high_weight,
            },
            "general": {
                "default": True,
                "initial_weight": 1.0 - initial_high_weight,
                "final_weight": 1.0 - final_high_weight,
            },
        },
    })


def resolve_group(source: str, config: dict) -> str:
    default_group = None
    for name, group in config["groups"].items():
        if group["default"]:
            default_group = name
        if source in group["sources"]:
            return name
    return default_group


class CurriculumMixtureSampler(Sampler[int]):
    """Grup ağırlıklarını eğitim adımına göre lineer değiştiren sampler."""

    def __init__(
        self,
        dataset,
        mixture_config: dict,
        seed: int = 42,
        samples_per_step: int = 1,
        num_samples: Optional[int] = None,
        chunk_size: int = 1024,
    ):
        self.config = validate_mixture_config(mixture_config)
        self.group_names = list(self.config["groups"])
        missing = [name for name in self.group_names if not dataset.group_ranges.get(name)]
        if missing:
            raise ValueError(f"Mixture gruplarında kullanılabilir blok yok: {missing}")
        self.group_ranges = {
            name: list(dataset.group_ranges[name]) for name in self.group_names
        }
        self.group_cumulative = {}
        self.group_sizes = {}
        for name, ranges in self.group_ranges.items():
            cumulative = []
            total = 0
            for start, end in ranges:
                total += end - start
                cumulative.append(total)
            self.group_cumulative[name] = cumulative
            self.group_sizes[name] = total
        self.seed = int(seed)
        self.samples_per_step = max(1, int(samples_per_step))
        self.num_samples = int(num_samples if num_samples is not None else len(dataset))
        if self.num_samples < 1:
            raise ValueError("Mixture sampler num_samples pozitif olmalı")
        self.chunk_size = max(1, int(chunk_size))
        self.epoch = 0
        self.start_step = 0

    def set_training_step(self, step: int) -> None:
        self.start_step = max(0, int(step))

    def weights_at(self, step: float) -> Dict[str, float]:
        curriculum_steps = self.config["curriculum_steps"]
        progress = 1.0 if curriculum_steps == 0 else min(max(step / curriculum_steps, 0.0), 1.0)
        raw = {
            name: group["initial_weight"]
            + (group["final_weight"] - group["initial_weight"]) * progress
            for name, group in self.config["groups"].items()
        }
        total = sum(raw.values())
        return {name: value / total for name, value in raw.items()}

    def _group_offset_to_index(self, group_name: str, offset: int) -> int:
        cumulative = self.group_cumulative[group_name]
        range_index = bisect_right(cumulative, offset)
        previous = cumulative[range_index - 1] if range_index else 0
        start, _ = self.group_ranges[group_name][range_index]
        return start + (offset - previous)

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        generator = torch.Generator().manual_seed(self.seed + epoch)
        emitted = 0
        while emitted < self.num_samples:
            count = min(self.chunk_size, self.num_samples - emitted)
            midpoint_sample = emitted + (count - 1) / 2
            step = self.start_step + midpoint_sample / self.samples_per_step
            weights = self.weights_at(step)
            weight_tensor = torch.tensor(
                [weights[name] for name in self.group_names], dtype=torch.float64
            )
            selected_groups = torch.multinomial(
                weight_tensor, count, replacement=True, generator=generator
            ).tolist()
            for group_index in selected_groups:
                name = self.group_names[group_index]
                offset = int(torch.randint(
                    self.group_sizes[name], (1,), generator=generator
                ).item())
                yield self._group_offset_to_index(name, offset)
            emitted += count

    def __len__(self):
        return self.num_samples

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "start_step": self.start_step}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.start_step = int(state.get("start_step", 0))
