"""Tests for midl._cache."""

from unittest.mock import MagicMock, patch

import pytest

from midl._cache import canonical_mhd, csv_url, ensure_cached


class TestCanonicalMhd:
    def test_mhd_32(self):
        assert canonical_mhd(32) == "mhd_032Re"

    def test_mhd_0(self):
        assert canonical_mhd(0) == "mhd_000Re"

    def test_mhd_180(self):
        assert canonical_mhd(180) == "mhd_180Re"

    def test_mhd_negative_boundary(self):
        assert canonical_mhd(-20) == "mhd_-20Re"

    def test_mhd_small_negative(self):
        assert canonical_mhd(-5) == "mhd_-05Re"

    def test_mhd_accepts_integer_float(self):
        assert canonical_mhd(32.0) == "mhd_032Re"

    def test_mhd_below_range(self):
        with pytest.raises(ValueError, match=r"\[-20, 180\]"):
            canonical_mhd(-21)

    def test_mhd_above_range(self):
        with pytest.raises(ValueError, match=r"\[-20, 180\]"):
            canonical_mhd(181)

    def test_mhd_non_integer(self):
        with pytest.raises(ValueError, match="must be an integer"):
            canonical_mhd(32.5)

    def test_mhd_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="must be an integer"):
            canonical_mhd("32")  # type: ignore[arg-type]


class TestCsvUrl:
    def test_32re(self):
        url = csv_url("2015-03", "32Re")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2015/03/201503_32Re.csv"

    def test_l1(self):
        url = csv_url("2024-01", "L1")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2024/01/202401_L1.csv"

    def test_mhd(self):
        url = csv_url("2024-05", "mhd_032Re")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2024/05/mhd/202405_mhd_032Re.csv"

    def test_mhd_zero(self):
        url = csv_url("2024-05", "mhd_000Re")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2024/05/mhd/202405_mhd_000Re.csv"


class TestEnsureCached:
    def test_downloads_on_miss(self, tmp_path):
        csv_content = b"timestamp,Bx\n2024-03-01T00:00:00,1.23\n"

        mock_resp = MagicMock()
        mock_resp.content = csv_content
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("midl._cache.cache_dir", return_value=tmp_path),
            patch("midl._cache.requests.get", return_value=mock_resp) as mock_get,
        ):
            path = ensure_cached("2024-03", "32Re")
            assert path.exists()
            assert path.read_bytes() == csv_content
            mock_get.assert_called_once()

    def test_serves_from_cache(self, tmp_path):
        cached_file = tmp_path / "202403_32Re.csv"
        cached_file.write_text("cached content")

        with (
            patch("midl._cache.cache_dir", return_value=tmp_path),
            patch("midl._cache.requests.get") as mock_get,
        ):
            path = ensure_cached("2024-03", "32Re")
            assert path == cached_file
            mock_get.assert_not_called()
