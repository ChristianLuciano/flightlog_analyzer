"""Tests for downsampling algorithms."""

import pytest
import numpy as np

from src.data.downsampling import (
    lttb_downsample, m4_downsample, simple_downsample,
    douglas_peucker, downsample
)
from src.core.types import DownsamplingMethod


class TestLTTBDownsampling:
    """Tests for LTTB algorithm."""

    def test_reduces_points(self):
        """Test that LTTB reduces number of points."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = lttb_downsample(x, y, 100)

        assert len(x_ds) == 100
        assert len(y_ds) == 100

    def test_preserves_endpoints(self):
        """Test that endpoints are preserved."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = lttb_downsample(x, y, 100)

        assert x_ds[0] == x[0]
        assert x_ds[-1] == x[-1]

    def test_no_downsampling_small_data(self):
        """Test no downsampling when data is small."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 2, 1])

        x_ds, y_ds = lttb_downsample(x, y, 100)

        assert len(x_ds) == 5

    def test_preserves_extremes(self):
        """Test that extremes are generally preserved."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x) + 0.01 * np.random.randn(1000)

        x_ds, y_ds = lttb_downsample(x, y, 200)

        # Downsampled should have similar range
        assert np.abs(y_ds.max() - y.max()) < 0.1
        assert np.abs(y_ds.min() - y.min()) < 0.1


class TestM4Downsampling:
    """Tests for M4 algorithm."""

    def test_reduces_points(self):
        """Test that M4 reduces number of points."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = m4_downsample(x, y, 100)

        assert len(x_ds) <= 100

    def test_preserves_extremes(self):
        """Test that extremes are preserved."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = m4_downsample(x, y, 100)

        # M4 should preserve min/max in each bucket
        assert np.isclose(y_ds.max(), y.max(), atol=0.1)
        assert np.isclose(y_ds.min(), y.min(), atol=0.1)


class TestDouglasPeucker:
    """Tests for Douglas-Peucker algorithm."""

    def test_simplifies_path(self):
        """Test that algorithm simplifies path."""
        # Circular path
        t = np.linspace(0, 2 * np.pi, 1000)
        x = np.cos(t)
        y = np.sin(t)

        x_ds, y_ds = douglas_peucker(x, y, epsilon=0.01)

        assert len(x_ds) < len(x)

    def test_preserves_endpoints(self):
        """Test that endpoints are preserved."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        x_ds, y_ds = douglas_peucker(x, y, epsilon=0.1)

        assert x_ds[0] == x[0]
        assert x_ds[-1] == x[-1]


class TestDownsampleFunction:
    """Tests for main downsample function."""

    def test_lttb_method(self):
        """Test LTTB method selection."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = downsample(x, y, 100, DownsamplingMethod.LTTB)
        assert len(x_ds) == 100

    def test_simple_method(self):
        """Test simple method selection."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        x_ds, y_ds = downsample(x, y, 100, DownsamplingMethod.SIMPLE)
        assert len(x_ds) <= 101  # May include extra endpoint

