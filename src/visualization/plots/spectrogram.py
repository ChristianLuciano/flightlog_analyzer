"""
Spectrogram plot implementation.

Provides time-frequency visualization using Short-Time Fourier Transform.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal as scipy_signal
from typing import Optional, Tuple

from ..base import BasePlot
from ...core.types import PlotConfig
from ...core.constants import DEFAULT_FFT_WINDOW_SIZE


class SpectrogramPlot(BasePlot):
    """
    Spectrogram for time-frequency analysis.

    Uses STFT to show frequency content evolution over time.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize SpectrogramPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.window_size = config.get("window_size", DEFAULT_FFT_WINDOW_SIZE)
        self.overlap = config.get("overlap", 0.75)
        self.window_type = config.get("window_type", "hann")
        self.colorscale = config.get("colorscale", "Viridis")
        self.show_db = config.get("show_db", True)

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render spectrogram.

        Args:
            data: DataFrame with timestamp and signal columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data
        self.figure = go.Figure()

        signals = self.config.get("signals", [])
        timestamp_col = self.config.get("x_axis", "timestamp")

        if not signals:
            signals = [
                col for col in data.columns if col != timestamp_col
            ][:1]

        if not signals:
            return self.figure

        signal_name = signals[0]
        if signal_name not in data.columns:
            return self.figure

        # Get signal data
        signal_data = data[signal_name].values
        timestamps = data[timestamp_col].values

        # Calculate sampling rate
        dt = np.median(np.diff(timestamps))
        fs = 1.0 / dt if dt > 0 else 1.0

        # Compute spectrogram
        times, freqs, Sxx = self._compute_spectrogram(signal_data, fs, timestamps)

        # Convert to dB if requested
        if self.show_db:
            Sxx = 10 * np.log10(Sxx + 1e-10)
            colorbar_title = "Power (dB)"
        else:
            colorbar_title = "Power"

        # Create heatmap
        self.figure.add_trace(go.Heatmap(
            x=times,
            y=freqs,
            z=Sxx,
            colorscale=self.colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "Time: %{x:.2f}<br>"
                "Freq: %{y:.2f} Hz<br>"
                "Power: %{z:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        # Layout
        layout = self._create_base_layout()
        layout.update({
            "title": self.title or f"Spectrogram - {self._get_signal_name(signal_name)}",
            "xaxis_title": "Time",
            "yaxis_title": "Frequency (Hz)",
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def _compute_spectrogram(
        self,
        signal: np.ndarray,
        fs: float,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using STFT.

        Args:
            signal: Input signal.
            fs: Sampling frequency.
            timestamps: Original timestamps.

        Returns:
            Tuple of (times, frequencies, spectrogram_matrix).
        """
        signal = np.nan_to_num(signal, nan=0.0)

        nperseg = min(self.window_size, len(signal))
        noverlap = int(nperseg * self.overlap)

        freqs, times, Sxx = scipy_signal.spectrogram(
            signal,
            fs=fs,
            window=self.window_type,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='spectrum'
        )

        # Map times back to original timestamp scale
        t_start = timestamps[0]
        t_end = timestamps[-1]
        times = t_start + times * (t_end - t_start) / times[-1] if times[-1] > 0 else times

        return times, freqs, Sxx

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        self.render(data)

    def set_frequency_range(self, f_min: float, f_max: float) -> None:
        """
        Set visible frequency range.

        Args:
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        if self.figure:
            self.figure.update_yaxes(range=[f_min, f_max])

    def _get_signal_name(self, path: str) -> str:
        """Extract display name from signal path."""
        parts = path.split(".")
        return parts[-1] if parts else path

