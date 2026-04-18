"""Download and local file-cache layer."""

from __future__ import annotations

from pathlib import Path

import platformdirs
import requests

BASE_URL = "https://csem.engin.umich.edu/MIDL/data"

MHD_VALID_RE: frozenset[int] = frozenset(range(-70, 71))


def canonical_mhd(target_re: float | int) -> str:
    """Validate and canonicalize an MHD target Re value.

    MHD data is only available at integer Re in ``[-70, 70]``.
    """
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
            f"MHD target_re must be an integer in [-70, 70], got {re_int}"
        )
    return f"mhd_{re_int:03d}Re"


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
