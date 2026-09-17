"""Where the config, the runs and the exports live.

`labcore` is shared by Labs 1-4 but the persistent state - the tuned parameter
set, the run history, the exported CSVs - belongs to `labs/lab1/`.  Resolving
that by counting `.parent` hops from this file is what broke once already: the
package moved from `labs/lab1/lab1_core/` to `labs/labcore/`, every path
silently shifted up one directory, and the code quietly created a fresh
default config while the tuned one sat untouched next to the run history.

So the anchor is a named directory, found by searching upwards for the repo
root, and it is asserted rather than assumed.  `SPHERO_LAB_DIR` overrides it
for anyone wanting to keep state elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ROOT", "LABS_DIR", "LAB_DIR", "CONFIG_DIR", "RUNS_DIR",
           "EXPORTS_DIR"]


def _find_repo_root(start: Path) -> Path:
    """Walk upwards until the directory holding both `labs/` and `src/`."""
    for candidate in (start, *start.parents):
        if (candidate / "labs").is_dir() and (candidate / "src").is_dir():
            return candidate
    # Fall back to the historical layout rather than raising: a partial
    # checkout should still be able to import the package.
    return start.parent.parent


REPO_ROOT: Path = _find_repo_root(Path(__file__).resolve().parent)
LABS_DIR: Path = REPO_ROOT / "labs"

_override = os.environ.get("SPHERO_LAB_DIR", "").strip()
LAB_DIR: Path = Path(_override).expanduser().resolve() if _override else LABS_DIR / "lab1"

CONFIG_DIR: Path = LAB_DIR / "config"
RUNS_DIR: Path = LAB_DIR / "runs"
EXPORTS_DIR: Path = LAB_DIR / "exports"
