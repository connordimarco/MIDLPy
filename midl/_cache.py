"""Download and local file-cache layer."""

from __future__ import annotations

from pathlib import Path

import platformdirs
import requests

BASE_URL = "https://csem.engin.umich.edu/MIDL/data"

TARGETS: dict[str, str] = {
    "14re": "14Re",
    "32re": "32Re",
    "l1": "L1",
}

MHD_VALID_RE: frozenset[int] = frozenset(range(-20, 181))


def _canonical_mhd(target_re: float | int) -> str:
    """Validate and canonicalize an MHD target Re value."""
    if not isinstance(target_re, (int, float)) or isinstance(target_re, bool):
        raise ValueError(
            f"MHD target_re must be an integer, got {type(target_re).__name__}"
        )
    as_float = float(target_re)
    if not as_float.is_integer():
        raise ValueError(f"MHD target_re must be an integer, got {target_re!r}")
    re_int = int(as_float)
    if re_int not in MHD_VALID_RE:
        raise ValueError(
            f"MHD target_re must be an integer in [-20, 180], got {re_int}"
        )
    return f"mhd_{re_int:03d}Re"


def resolve_target(target: str, target_re: float | int | None = None) -> str:
    """Normalize a case-insensitive target string to its canonical form.

    Parameters
    ----------
    target : str
        ``"l1"``, ``"14re"``, ``"32re"``, or ``"mhd"`` (case-insensitive).
    target_re : int, optional
        Required when ``target="mhd"``. Must be an integer in ``[-20, 180]``.
        Must be ``None`` for all other targets.
    """
    key = target.lower()
    if key == "mhd":
        if target_re is None:
            raise ValueError(
                "target='mhd' requires target_re (an integer in [-20, 180])"
            )
        return _canonical_mhd(target_re)

    if target_re is not None:
        raise ValueError(
            f"target_re is only valid with target='mhd', not target={target!r}"
        )
    canonical = TARGETS.get(key)
    if canonical is None:
        valid = ", ".join([*TARGETS.values(), "mhd"])
        raise ValueError(f"Unknown target {target!r}. Valid targets: {valid}")
    return canonical


def csv_url(year_month: str, target: str) -> str:
    """Build the full URL for a monthly CSV file.

    Parameters
    ----------
    year_month : str
        ``"YYYY-MM"`` string.
    target : str
        Canonical target name (``"14Re"``, ``"32Re"``, ``"L1"``, or
        ``"mhd_NNNRe"``).
    """
    year, month = year_month.split("-")
    if target.startswith("mhd_"):
        return f"{BASE_URL}/{year}/{month}/mhd/{year}{month}_{target}.csv"
    return f"{BASE_URL}/{year}/{month}/{year}{month}_{target}.csv"


def cache_dir() -> Path:
    """Return (and create) the local cache directory."""
    d = Path(platformdirs.user_cache_dir("midl"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_cached(year_month: str, target: str) -> Path:
    """Return a local path to the CSV, downloading it if not already cached."""
    year, month = year_month.split("-")
    filename = f"{year}{month}_{target}.csv"
    path = cache_dir() / filename

    if path.exists():
        return path

    url = csv_url(year_month, target)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path
