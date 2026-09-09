"""The core library must stay importable without torch, a GPU, or a database.

CLAUDE.md states this under "Layering" and nothing enforced it. Phase 3 made it easy to
break: `tayr.config` gained a device key, `tayr.train` and `tayr.eval.harness` are new,
and a single module-level `import torch` in any of them would make `tayr config validate`
require a two-gigabyte install to parse a YAML file.

The check blocks the heavy packages at import time rather than reasoning about the source,
so it catches an indirect import through a third module as well as a direct one.
"""

from __future__ import annotations

import subprocess
import sys

#: Modules that must import with nothing from the `cv` extra present.
CORE_MODULES = (
    "tayr.cli.main",
    "tayr.config",
    "tayr.datasets.coco",
    "tayr.determinism",
    "tayr.devices",
    "tayr.errors",
    "tayr.eval",
    "tayr.eval.harness",
    "tayr.geometry",
    "tayr.manifest",
    # Draws with cv2 and encodes with av, both imported inside the functions that use
    # them - so `tayr watch run` without --render never pays for either.
    "tayr.render.annotate",
    "tayr.train",
    "tayr.train.detector",
)

#: Blocked during the check. `torch` and `rfdetr` are the `cv` extra's weight; `cv2` and
#: `torchvision` come with it; `pytorch_lightning` arrives through RF-DETR's trainer.
HEAVY = ("torch", "torchvision", "rfdetr", "cv2", "pytorch_lightning")

_SCRIPT = """
import importlib, sys

BLOCKED = set({heavy!r})

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(name.split(".")[0] + " is blocked by the layering check")
        return None

sys.meta_path.insert(0, Blocker())

failures = []
for module in {modules!r}:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        failures.append(module + ": " + str(exc))

if failures:
    print("\\n".join(failures))
    raise SystemExit(1)
"""


def test_core_modules_import_without_the_cv_extra() -> None:
    """Run in a subprocess: this process has already imported torch via other tests."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(heavy=HEAVY, modules=CORE_MODULES)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "these modules need the cv extra just to import, which breaks `tayr config "
        f"validate` on a machine without it:\n{result.stdout}{result.stderr}"
    )


def test_the_blocker_actually_blocks() -> None:
    """A guard whose failure mode is a silent pass is not a guard."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(heavy=HEAVY, modules=("torch",))],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    assert "blocked by the layering check" in result.stdout
