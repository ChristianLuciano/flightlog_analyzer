"""
FFT plot implementation.

Provides frequency domain visualization including FFT magnitude,
power spectral density, and phase plots.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal as scipy_signal
from typing import Dict, Any, Optional, List, Tuple

from ..base import BasePlot
from ...core.types import PlotConfig
from ...core.constants import (
    DEFAULT_FFT_WINDOW_SIZE,
    DEFAULT_FFT_OVERLAP,
    DEFAULT_FFT_WINDOW_TYPE
)


class FFTPlot(BasePlot):
    """
    FFT frequency domain plot.

    Supports magnitude spectrum, power spectral density,
    and configurable windowing functions.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize FFTPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.window_size = config.get("window_size", DEFAULT_FFT_WINDOW_SIZE)
        self.overlap = config.get("overlap", DEFAULT_FFT_OVERLAP)
        self.window_type = config.get("window_type", DEFAULT_FFT_WINDOW_TYPE)
        self.show_db = config.get("show_db", True)
        self.show_phase = config.get("show_phase", False)

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render FFT plot.

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
            ][:1]  # Default to first signal

        colors = self._get_color_sequence()
        
        # Default y_label in case no signals are processed
        y_label = "Magnitude (dB)" if self.show_db else "Magnitude"

        for i, signal_name in enumerate(signals):
            if signal_name not in data.columns:
                continue

            # Get signal data
            signal_data = data[signal_name].values
            timestamps = data[timestamp_col].values

            # Calculate sampling rate
            dt = np.median(np.diff(timestamps))
            fs = 1.0 / dt if dt > 0 else 1.0

            # Compute FFT
            freqs, magnitude, phase = self._compute_fft(signal_data, fs)

            # Convert to dB if requested
            if self.show_db:
                magnitude = 20 * np.log10(magnitude + 1e-10)
                y_label = "Magnitude (dB)"
            else:
                y_label = "Magnitude"

            # Add magnitude trace
            self.figure.add_trace(go.Scatter(
                x=freqs,
                y=magnitude,
                mode='lines',
                name=f"{self._get_signal_name(signal_name)}",
                line=dict(
                    color=colors[i % len(colors)],
                    width=1.5
                ),
                hovertemplate=(
                    f"<b>{signal_name}</b><br>"
                    "Freq: %{x:.2f} Hz<br>"
                    "Mag: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

        # Apply layout
        layout = self._create_base_layout()
        layout.update({
            "xaxis_title": "Frequency (Hz)",
            "yaxis_title": y_label,
            "xaxis_type": "log" if self.config.get("log_freq", False) else "linear",
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def _compute_fft(
        self,
        signal: np.ndarray,
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FFT of signal.

        Args:
            signal: Input signal array.
            fs: Sampling frequency.

        Returns:
            Tuple of (frequencies, magnitude, phase).
        """
        # Remove NaN values
        signal = np.nan_to_num(signal, nan=0.0)

        n = len(signal)

        # Apply window
        window = self._get_window(n)
        windowed_signal = signal * window

        # Compute FFT
        fft_result = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(n, 1.0 / fs)

        # Compute magnitude and phase
        magnitude = np.abs(fft_result) * 2.0 / n
        phase = np.angle(fft_result)

        return freqs, magnitude, phase

    def _get_window(self, n: int) -> np.ndarray:
        """Get window function."""
        window_functions = {
            "rectangular": np.ones,
            "hanning": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
            "bartlett": np.bartlett,
        }

        func = window_functions.get(self.window_type.lower(), np.hanning)
        return func(n)

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        # Re-render for FFT as it requires full recomputation
        self.render(data)

    def compute_psd(
        self,
        signal: np.ndarray,
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.

        Args:
            signal: Input signal.
            fs: Sampling frequency.

        Returns:
            Tuple of (frequencies, PSD values).
        """
        signal = np.nan_to_num(signal, nan=0.0)

        nperseg = min(self.window_size, len(signal))
        noverlap = int(nperseg * self.overlap)

        freqs, psd = scipy_signal.welch(
            signal,
            fs=fs,
            window=self.window_type,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )

        return freqs, psd

    def add_peak_markers(
        self,
        n_peaks: int = 5,
        min_prominence: float = 0.1
    ) -> None:
        """
        Add markers at spectral peaks.

        Args:
            n_peaks: Maximum number of peaks to mark.
            min_prominence: Minimum peak prominence.
        """
        if not self.figure or not self.figure.data:
            return

        # Get magnitude data from first trace
        freqs = np.array(self.figure.data[0].x)
        magnitude = np.array(self.figure.data[0].y)

        # Find peaks
        peaks, properties = scipy_signal.find_peaks(
            magnitude,
            prominence=min_prominence
        )

        if len(peaks) == 0:
            return

        # Sort by magnitude and take top N
        peak_mags = magnitude[peaks]
        sorted_indices = np.argsort(peak_mags)[::-1][:n_peaks]
        top_peaks = peaks[sorted_indices]

        # Add markers
        self.figure.add_trace(go.Scatter(
            x=freqs[top_peaks],
            y=magnitude[top_peaks],
            mode='markers+text',
            name='Peaks',
            marker=dict(
                color='#e74c3c',
                size=10,
                symbol='diamond',
            ),
            text=[f"{f:.1f} Hz" for f in freqs[top_peaks]],
            textposition='top center',
        ))

    def add_frequency_band(
        self,
        f_low: float,
        f_high: float,
        name: str = "",
        color: str = "rgba(255, 200, 0, 0.2)"
    ) -> None:
        """
        Highlight frequency band.

        Args:
            f_low: Lower frequency bound.
            f_high: Upper frequency bound.
            name: Band name for annotation.
            color: Fill color.
        """
        if not self.figure:
            return

        self.figure.add_vrect(
            x0=f_low,
            x1=f_high,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=name,
            annotation_position="top",
        )

    def _get_signal_name(self, path: str) -> str:
        """Extract display name from signal path."""
        parts = path.split(".")
        return parts[-1] if parts else path

