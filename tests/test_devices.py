"""Device resolution.

The rule under test throughout: `auto` may fall back and must record that it did; an
explicit accelerator must never silently become something else.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from tayr import devices
from tayr.devices import ResolvedDevice, resolve_device, validate_device_spec
from tayr.errors import ConfigError, DependencyUnavailableError


class FakeCuda:
    def __init__(self, *, available: bool, count: int = 1, capability: tuple[int, int] = (8, 0)):
        self._available, self._count, self._capability = available, count, capability

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int = 0) -> str:
        return f"FakeGPU-{index}"

    def get_device_capability(self, index: int = 0) -> tuple[int, int]:
        del index  # one fake GPU; every index reports the same capability
        return self._capability

    def get_arch_list(self) -> list[str]:
        return ["sm_75", "sm_80", "sm_86", "sm_90"]


def fake_torch(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=FakeCuda(**kwargs),
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_torch", lambda: fake_torch(available=False))


@pytest.fixture
def with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_torch", lambda: fake_torch(available=True))


class TestSpecValidation:
    @pytest.mark.parametrize("spec", ["auto", "cpu", "cuda", "cuda:0", "cuda:3", "mps"])
    def test_accepts_every_documented_form(self, spec: str) -> None:
        assert validate_device_spec(spec) == spec

    @pytest.mark.parametrize("spec", ["gpu", "CUDA", "cuda:", "cuda:x", "tpu", ""])
    def test_rejects_anything_else(self, spec: str) -> None:
        with pytest.raises(ConfigError, match="not a recognised device"):
            validate_device_spec(spec)

    def test_validation_does_not_need_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`tayr config validate` must work on a machine with no cv extra installed."""

        def explode() -> Any:
            raise AssertionError("validate_device_spec must not import torch")

        monkeypatch.setattr(devices, "_torch", explode)
        assert validate_device_spec("cuda:1") == "cuda:1"


class TestAutoMayFallBack:
    @pytest.mark.usefixtures("with_cuda")
    def test_auto_takes_cuda_when_present(self) -> None:
        resolved = resolve_device("auto")
        assert resolved.device == "cuda"
        assert resolved.fell_back is False

    @pytest.mark.usefixtures("no_cuda")
    def test_auto_falls_back_to_cpu_and_says_so(self) -> None:
        resolved = resolve_device("auto")
        assert resolved.device == "cpu"
        assert resolved.fell_back is True
        assert "CPU" in resolved.note


class TestExplicitRequestsMayNot:
    @pytest.mark.usefixtures("no_cuda")
    def test_cuda_without_cuda_raises_rather_than_using_cpu(self) -> None:
        """The failure this prevents is a 40x-slower run reported as though it were a GPU run."""
        with pytest.raises(DependencyUnavailableError, match="Refusing to fall back"):
            resolve_device("cuda")

    def test_cuda_index_beyond_the_device_count_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(devices, "_torch", lambda: fake_torch(available=True, count=1))
        with pytest.raises(ConfigError, match=re.escape("valid indices 0..0")):
            resolve_device("cuda:1")

    @pytest.mark.usefixtures("no_cuda")
    def test_mps_without_mps_raises(self) -> None:
        with pytest.raises(DependencyUnavailableError, match="Metal"):
            resolve_device("mps")

    @pytest.mark.usefixtures("with_cuda")
    def test_cpu_is_honoured_even_when_a_gpu_is_present(self) -> None:
        resolved = resolve_device("cpu")
        assert resolved.device == "cpu"
        assert resolved.fell_back is False
        assert resolved.cuda_available is True


class TestArchitectureGuard:
    """A GPU older than every architecture in the wheel fails at the first kernel launch.

    That is minutes into a run, after the dataloader is warm, with an error that reads
    like a CUDA install problem. Catching it up front is the difference between a
    ten-second failure and a wasted GPU hour.
    """

    def test_a_gpu_older_than_the_wheel_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sm_60 (Pascal, e.g. Tesla P100) against a wheel built for sm_75 and up.
        monkeypatch.setattr(
            devices, "_torch", lambda: fake_torch(available=True, capability=(6, 0))
        )
        with pytest.raises(DependencyUnavailableError) as excinfo:
            resolve_device("cuda")
        message = str(excinfo.value)
        assert "sm_60" in message
        assert "no kernels" in message

    def test_a_supported_gpu_records_its_capability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            devices, "_torch", lambda: fake_torch(available=True, capability=(8, 6))
        )
        resolved = resolve_device("cuda:0")
        assert resolved.compute_capability == "sm_86"
        assert "sm_86" in resolved.supported_architectures

    def test_a_gpu_newer_than_the_wheel_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA can JIT from embedded PTX, so newer-than-compiled is torch's call, not ours."""
        monkeypatch.setattr(
            devices, "_torch", lambda: fake_torch(available=True, capability=(12, 0))
        )
        assert resolve_device("cuda").compute_capability == "sm_120"


@pytest.mark.usefixtures("with_cuda")
def test_resolved_device_serialises_for_the_manifest() -> None:
    payload = resolve_device("auto").to_dict()
    assert payload["device"] == "cuda"
    assert set(payload) == {f.name for f in ResolvedDevice.__dataclass_fields__.values()}
