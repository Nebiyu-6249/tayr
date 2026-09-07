"""Device selection, resolved once and recorded.

The same config has to run on a Kaggle P100 and on a laptop with no GPU, so the device
is a config key, not a flag. Two rules govern what happens when the request cannot be
met:

**`auto` may fall back. An explicit request may not.** ``device: auto`` means "pick for
me" and resolving to CPU is a correct answer. ``device: cuda`` is a statement about the
machine, and if there is no CUDA there, silently training on CPU is the failure mode
this project exists to avoid - a run that takes forty times longer and is reported as
though it were the GPU run. That raises.

**What was actually chosen is recorded.** :class:`ResolvedDevice` goes into the run
manifest, so a reader can tell whether a number came from a GPU or from a CPU fallback
without having to trust the config file next to it.

**The GPU is also checked against the wheel that will drive it.** A torch wheel only
contains kernels for the architectures it was compiled for, and a GPU older than all of
them fails at the first kernel launch, not at import - which is to say, after the data
loader is warm and several minutes into a run. `torch 2.13.0` as resolved from PyPI's
default index is a CUDA 13.0 build whose compiled architectures are
`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`
`[VERIFIED: python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
-> 13.0 [...]`, in this session]. Anything below sm_75 - Pascal and Volta among them -
has no kernels in that wheel. :func:`resolve_device` compares the two up front and says
so.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from tayr.errors import ConfigError, DependencyUnavailableError

#: Accepted values for the `device` config key. Validated without importing torch so
#: `tayr config validate` works on a machine that has no CV extra installed.
DEVICE_PATTERN = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    """The device a run actually got, and how it got there."""

    requested: str
    """Verbatim from the config."""
    device: str
    """Concrete torch device string. Never `auto`."""
    accelerator: str
    """Human-readable family: cpu, cuda or mps. Recorded so a manifest reader does not
    have to parse `device`."""
    cuda_available: bool
    device_name: str | None
    """Reported by torch for CUDA devices, e.g. 'Tesla P100-PCIE-16GB'. None otherwise."""
    fell_back: bool
    """True when `auto` could not get an accelerator. A CPU number and a GPU number are
    not comparable, and this is the flag that says which one you are reading."""
    compute_capability: str | None = None
    """`sm_XX` for the selected CUDA device. None for CPU and MPS."""
    supported_architectures: tuple[str, ...] = ()
    """What the installed torch wheel was actually compiled for."""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailableError(
            "torch is required to resolve a device but is not installed. "
            "Install the 'cv' extra: pip install -e '.[cv]'"
        ) from exc
    return torch


def _architecture_check(torch: Any, index: int) -> tuple[str, tuple[str, ...]]:
    """Confirm the installed torch wheel has kernels for this GPU.

    Returns `(compute_capability, supported_architectures)`. Raises when the GPU is
    older than everything the wheel was compiled for, because the alternative is a run
    that imports cleanly, loads the dataset, and then dies at the first matmul with
    "no kernel image is available for execution on the device".

    A GPU *newer* than the wheel's highest architecture is not an error: CUDA's forward
    compatibility can JIT from the embedded PTX, so that case is left to torch.
    """
    major, minor = torch.cuda.get_device_capability(index)
    capability = f"sm_{major}{minor}"
    supported = tuple(str(a) for a in torch.cuda.get_arch_list())
    numeric = [
        int(a.removeprefix("sm_")) for a in supported if a.startswith("sm_") and a[3:].isdigit()
    ]
    if numeric and (major * 10 + minor) < min(numeric):
        raise DependencyUnavailableError(
            f"CUDA device {index} ({torch.cuda.get_device_name(index)}) is {capability}, "
            f"but this torch build ({torch.__version__}, CUDA {torch.version.cuda}) was "
            f"compiled only for {', '.join(supported)}. There are no kernels for this GPU "
            "and the run would fail at the first kernel launch, minutes in. Install a "
            "torch build that covers this architecture, or use a newer GPU."
        )
    return capability, supported


def validate_device_spec(spec: str) -> str:
    """Check the shape of a device string without needing torch installed."""
    if not DEVICE_PATTERN.match(spec):
        raise ConfigError(
            f"device={spec!r} is not a recognised device. Use 'auto', 'cpu', 'cuda', "
            "'cuda:<index>' or 'mps'."
        )
    return spec


def resolve_device(spec: str) -> ResolvedDevice:
    """Turn a config device spec into the concrete device a run will use.

    Raises `DependencyUnavailableError` when an accelerator is named explicitly and is
    not present. Only `auto` is allowed to come back with something other than what was
    asked for, and when it does, `fell_back` records it.
    """
    validate_device_spec(spec)
    torch = _torch()
    cuda_available = bool(torch.cuda.is_available())

    if spec == "auto":
        if cuda_available:
            capability, supported = _architecture_check(torch, 0)
            return ResolvedDevice(
                requested=spec,
                device="cuda",
                accelerator="cuda",
                cuda_available=True,
                device_name=str(torch.cuda.get_device_name(0)),
                fell_back=False,
                compute_capability=capability,
                supported_architectures=supported,
                note="auto resolved to CUDA",
            )
        return ResolvedDevice(
            requested=spec,
            device="cpu",
            accelerator="cpu",
            cuda_available=False,
            device_name=None,
            fell_back=True,
            note=(
                "auto found no CUDA device and fell back to CPU. Any timing or "
                "throughput number from this run describes CPU execution."
            ),
        )

    if spec == "cpu":
        return ResolvedDevice(
            requested=spec,
            device="cpu",
            accelerator="cpu",
            cuda_available=cuda_available,
            device_name=None,
            fell_back=False,
            note=(
                "CPU requested explicitly while a CUDA device was present"
                if cuda_available
                else "CPU requested explicitly"
            ),
        )

    if spec == "mps":
        if not bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise DependencyUnavailableError(
                "device='mps' was requested but Metal Performance Shaders are not "
                "available here. Use 'cpu', or 'auto' to let the run choose."
            )
        return ResolvedDevice(
            requested=spec,
            device="mps",
            accelerator="mps",
            cuda_available=cuda_available,
            device_name=None,
            fell_back=False,
            note="MPS requested explicitly",
        )

    # cuda or cuda:N
    if not cuda_available:
        raise DependencyUnavailableError(
            f"device={spec!r} was requested but torch reports no CUDA device "
            f"(torch {torch.__version__}). Refusing to fall back to CPU: a CPU run is "
            "roughly two orders of magnitude slower and its numbers are not comparable "
            "to a GPU run. Set device: auto to allow a CPU fallback deliberately."
        )

    index = int(spec.split(":", 1)[1]) if ":" in spec else 0
    count = int(torch.cuda.device_count())
    if index >= count:
        raise ConfigError(
            f"device={spec!r} names CUDA device {index}, but torch reports only "
            f"{count} device(s) (valid indices 0..{count - 1})."
        )

    capability, supported = _architecture_check(torch, index)
    return ResolvedDevice(
        requested=spec,
        device=spec,
        accelerator="cuda",
        cuda_available=True,
        device_name=str(torch.cuda.get_device_name(index)),
        fell_back=False,
        compute_capability=capability,
        supported_architectures=supported,
        note=f"CUDA device {index} requested explicitly",
    )
