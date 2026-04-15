"""Tests for midl._propagate."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from midl._loader import load
from midl._propagate import _ballistic_propagate_df, propagate

DATA_DIR = Path(__file__).parent / "data"


def _synthetic_l1_df(n=10, ux=-400.0, x_re=200.0):
    idx = pd.date_range("2024-05-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "Bx": np.arange(n, dtype=float),
            "By": np.zeros(n),
            "Bz": np.zeros(n),
            "Ux": np.full(n, ux),
            "Uy": np.zeros(n),
            "Uz": np.zeros(n),
            "rho": np.full(n, 5.0),
            "T": np.full(n, 1e5),
            "X": np.full(n, x_re),
        },
        index=idx,
    )


class TestBallisticDf:
    def test_constant_wind_delay_matches_analytic(self):
        df = _synthetic_l1_df(n=60, ux=-400.0, x_re=200.0)
        result = _ballistic_propagate_df(df, target_re=14.0)
        assert len(result) == len(df)
        assert list(result.columns) == [
            "Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T",
        ]
        # Analytic delay: (200 - 14) * 6371 km / 400 km/s ≈ 2963 s ≈ 49 min.
        # At t=55 min, the value should equal the input Bx from t ≈ 6 min,
        # which is 6.0. Allow a wide tolerance because of the limit=2
        # interpolation bridge + integer-second rounding.
        delay_min = round((200 - 14) * 6371 / 400 / 60)
        probe = result.index[-1]
        source_time = probe - pd.Timedelta(minutes=delay_min)
        # probe should be near where source was
        expected_bx = df.loc[source_time, "Bx"]
        assert abs(result.loc[probe, "Bx"] - expected_bx) <= 1.5

    def test_grid_matches_input_range(self):
        df = _synthetic_l1_df(n=30)
        result = _ballistic_propagate_df(df, target_re=14.0)
        assert result.index[0] == df.index[0]
        assert result.index[-1] == df.index[-1]
        assert len(result) == len(df)

    def test_drops_source_columns(self):
        df = _synthetic_l1_df(n=10)
        df["B_source"] = "13"
        result = _ballistic_propagate_df(df, target_re=14.0)
        assert "B_source" not in result.columns
        assert "X" not in result.columns

    def test_nan_row_does_not_poison_neighbors(self):
        """A NaN-X row must not invalidate the causality check for earlier rows."""
        df = _synthetic_l1_df(n=60, ux=-400.0, x_re=200.0)
        # Zero out row 30 across the board, mimicking a data gap.
        df.iloc[30] = np.nan
        result = _ballistic_propagate_df(df, target_re=14.0)
        # A probe near the end should receive the Bx value from the
        # corresponding earlier source minute (~49 min upstream), not NaN.
        delay_min = round((200 - 14) * 6371 / 400 / 60)
        probe = result.index[-1]
        source_time = probe - pd.Timedelta(minutes=delay_min)
        assert not np.isnan(result.loc[probe, "Bx"])
        assert abs(result.loc[probe, "Bx"] - df.loc[source_time, "Bx"]) <= 1.5


class TestPropagate:
    def _load_l1(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"):
            return load("2024-03-01 00:00", "2024-03-01 00:09", "l1")

    def test_ballistic_returns_dataset(self):
        ds = self._load_l1()
        result = propagate(ds, "ballistic", 14)
        assert isinstance(result, xr.Dataset)
        assert "Bx" in result.data_vars
        assert "X" not in result.data_vars
        assert result.attrs["midl_propagation"] == {"method": "ballistic", "target_re": 14.0}
        assert result.attrs["target"] == "14Re"

    def test_preserves_variable_attrs(self):
        ds = self._load_l1()
        result = propagate(ds, "ballistic", 14)
        assert result["Bx"].attrs["units"] == "nT"
        assert result["Ux"].attrs["coordinate_system"] == "GSM"

    def test_mhd_rejected_as_unknown(self):
        ds = self._load_l1()
        with pytest.raises(ValueError, match="Unknown method"):
            propagate(ds, "mhd", 14)

    def test_unknown_method(self):
        ds = self._load_l1()
        with pytest.raises(ValueError, match="Unknown method"):
            propagate(ds, "magic", 14)

    def test_method_case_insensitive(self):
        ds = self._load_l1()
        result = propagate(ds, "BALLISTIC", 14)
        assert result.attrs["midl_propagation"]["method"] == "ballistic"

    def test_requires_x_column(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:09", 32)
        with pytest.raises(ValueError, match="requires the L1 dataset"):
            propagate(ds, "ballistic", 14)

    def test_warns_when_already_propagated(self):
        ds = self._load_l1()
        once = propagate(ds, "ballistic", 14)
        # Re-attach X so the X-check passes and we can exercise the warning path.
        once = once.assign(X=ds["X"])
        with pytest.warns(UserWarning, match="already-propagated"):
            propagate(once, "ballistic", 32)

    def test_accessor(self):
        ds = self._load_l1()
        via_accessor = ds.midl.propagate("ballistic", 14)
        via_function = propagate(ds, "ballistic", 14)
        xr.testing.assert_equal(via_accessor, via_function)


class TestLoaderTagging:
    def test_l1_tag_is_none(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:09", "l1")
        assert ds.attrs["midl_propagation"] is None

    def test_32re_tag(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:09", 32)
        assert ds.attrs["midl_propagation"] == {"method": "ballistic", "target_re": 32.0}
