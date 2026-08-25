"""Seeding and reproducibility.

Honest scope, stated up front because overclaiming it in a dissertation would be a
real problem: bit-exact reproducibility on GPU is NOT generally achievable. Several
cuDNN kernels and any op using atomic accumulation are nondeterministic by design, and
results can also shift with batch size, driver, and library version.

What `seed_everything` actually guarantees:
  - identical hardware + identical pinned versions + identical seed -> bit-exact
  - anything else -> close, but not bitwise identical

The run manifest records the seed and every version so a reader can tell which case
they are in. See docs/RESEARCH.md section 8.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedReport:
    """What seeding actually managed to do, recorded verbatim in the run manifest."""

    seed: int
    numpy_seeded: bool
    python_seeded: bool
    torch_seeded: bool
    torch_deterministic_algorithms: bool
    cuda_available: bool
    note: str


def seed_everything(seed: int, *, strict_torch: bool = False) -> SeedReport:
    """Seed Python, NumPy and (if installed) torch.

    torch is an optional dependency, so its absence is reported rather than raised -
    the core library must stay importable without it. Pass `strict_torch=True` from
    training entry points, where a missing torch means the run cannot proceed and
    silently continuing would be the failure mode this codebase exists to avoid.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32); got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    # Makes hash-order-dependent iteration reproducible in child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_seeded = False
    deterministic = False
    cuda = False
    note = "torch not installed; CPU-only seeding applied"

    try:
        import torch
    except ImportError:
        if strict_torch:
            from tayr.errors import DependencyUnavailableError

            raise DependencyUnavailableError(
                "torch is required for this operation but is not installed. "
                "Install the 'cv' extra: pip install -e '.[cv]'"
            ) from None
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch_seeded = True
        cuda = bool(torch.cuda.is_available())
        # cuBLAS needs this set before init for deterministic GEMMs on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic = True
            note = "torch seeded; deterministic algorithms requested (warn_only)"
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            note = f"torch seeded; deterministic algorithms unavailable: {exc}"

    return SeedReport(
        seed=seed,
        numpy_seeded=True,
        python_seeded=True,
        torch_seeded=torch_seeded,
        torch_deterministic_algorithms=deterministic,
        cuda_available=cuda,
        note=note,
    )
