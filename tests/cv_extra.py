"""Skip marker for tests that need the `cv` extra.

CI's default job installs `.[dev]` only - torch and rfdetr together run to gigabytes and
pulling them into every lint-and-test run would be indefensible. These tests therefore
skip where the extra is absent and run where it is present.

That is a real coverage gap, not a free lunch: the whole point of
`test_train_detector.py::TestUpstreamContract` is to fail when RF-DETR renames a field,
and a test that silently skips catches nothing. The `cv-contract` job in
`.github/workflows/ci.yml` exists to make sure it actually runs somewhere.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_CV_EXTRA = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "rfdetr", "cv2")
)

requires_cv_extra = pytest.mark.skipif(
    not HAS_CV_EXTRA,
    reason="needs the cv extra (torch, rfdetr, opencv): pip install -e '.[cv]'",
)
