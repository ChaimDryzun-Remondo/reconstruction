"""Shared bootstrap for smoke tests that load demo scripts importing
from ``RemondoPythonCore.Common.*`` and
``RemondoPythonCore.external_reconstruction.*``.

Two conditions, each individually sufficient, would block a real
``RemondoPythonCore.Common.*`` import from succeeding under the
submodule's pytest:

1. The submodule's pytest does not have ``c:/git/`` (the parent of
   ``RemondoPythonCore/``) on its python path.
2. The ``conftest.py`` ``RemondoPythonCore.*`` mock installation
   (lines that begin at the ``_install_mocks`` definition) populates
   empty-path namespace stubs in ``sys.modules`` so that the standalone
   core profile can run without the parent monorepo.  The same stubs
   pollute ``sys.modules`` for the monorepo-profile smoke tests that
   want the real package.

Sprint 4 thread 3 introduces two smoke tests
(``test_example_flow_smoke.py`` and ``test_example_smoke.py``) that
need to load demo scripts via ``importlib.util.spec_from_file_location``.
This helper centralises the bootstrap they need so the cleanup logic
is not duplicated across the two callers.

Sprint 5 item 3 removed the ``Shared.Common.*`` mock siblings that the
original docstring named as the cache-pollution source.  The motivating
``RemondoPythonCore.*`` mocks remain in place — they are required by
the standalone core profile — so this helper continues to be required
even after the ``Shared.Common`` cleanup.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def bootstrap_remondopythoncore() -> None:
    """Prepare the runtime so ``import RemondoPythonCore.<anything>`` resolves
    against the expected on-disk location.

    Order of operations (matters for correctness):

    1. Drop stale ``RemondoPythonCore.*`` entries from ``sys.modules``.
       Done first so a re-import attempt cannot find the cached stale
       entry before cleanup runs.
    2. Invalidate ``importlib`` finder caches.
    3. Ensure the expected parent directory (``c:/git/``) is at the
       front of ``sys.path``.
    4. Re-import ``RemondoPythonCore`` cleanly.
    5. Verify the import resolved to the expected on-disk location;
       raise a clear ``RuntimeError`` if not.

    The post-import verification catches the failure mode where the
    bootstrap superficially succeeds (``import RemondoPythonCore`` does
    not raise) but ``RemondoPythonCore`` resolves from an unexpected
    location -- e.g. a sibling installation on ``sys.path`` or a leaked
    namespace-package fragment.  The clear error here is preferable to
    a confusing failure later in the smoke test.
    """
    expected_parent = Path(__file__).resolve().parents[3]
    expected_remondo_dir = (expected_parent / "RemondoPythonCore").resolve()

    # 1. Drop stale RemondoPythonCore.* entries.
    stale = [
        k for k in list(sys.modules)
        if k == "RemondoPythonCore" or k.startswith("RemondoPythonCore.")
    ]
    for k in stale:
        del sys.modules[k]

    # 2. Invalidate importlib finder caches.
    importlib.invalidate_caches()

    # 3. Ensure expected parent at front of sys.path.
    parent_str = str(expected_parent)
    if not sys.path or sys.path[0] != parent_str:
        if parent_str in sys.path:
            sys.path.remove(parent_str)
        sys.path.insert(0, parent_str)

    # 4. Re-import RemondoPythonCore cleanly.
    import RemondoPythonCore  # noqa: F401 — imported to populate sys.modules

    # 5. Verify the import resolved to the expected on-disk location.
    actual_paths = [Path(p).resolve() for p in RemondoPythonCore.__path__]
    if expected_remondo_dir not in actual_paths:
        raise RuntimeError(
            "RemondoPythonCore bootstrap failed: expected to load from "
            f"{expected_remondo_dir} but RemondoPythonCore.__path__ is "
            f"{list(RemondoPythonCore.__path__)}.  The bootstrap helper "
            "may need updating if the repo layout has changed, or there "
            "may be a conflicting RemondoPythonCore on sys.path."
        )
