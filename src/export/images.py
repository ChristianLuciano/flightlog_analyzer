"""
Image export functionality.

Provides export of plots and screenshots to various image formats.
"""

import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Union
import logging

from ..core.exceptions import ExportError

logger = logging.getLogger(__name__)


def export_plot_image(
    figure: go.Figure,
    path: Union[str, Path],
    format: str = 'png',
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0
) -> None:
    """
    Export Plotly figure to image file.

    Args:
        figure: Plotly figure.
        path: Output path.
        format: Image format ('png', 'svg', 'pdf', 'jpeg').
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Scale factor for resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format.lower() == 'svg':
            figure.write_image(
                str(path),
                format='svg',
                width=width,
                height=height
            )
        else:
            figure.write_image(
                str(path),
                format=format,
                width=width,
                height=height,
                scale=scale
            )

        logger.info(f"Exported plot to {path}")

    except Exception as e:
        raise ExportError(f"Failed to export image: {e}")


def export_screenshot(
    figure: go.Figure,
    path: Union[str, Path],
    **kwargs
) -> None:
    """
    Export figure as screenshot.

    Convenience wrapper for export_plot_image.
    """
    export_plot_image(figure, path, **kwargs)


def figures_to_pdf(
    figures: list,
    path: Union[str, Path],
    width: int = 800,
    height: int = 600
) -> None:
    """
    Export multiple figures to a single PDF.

    Args:
        figures: List of Plotly figures.
        path: Output PDF path.
        width: Image width.
        height: Image height.
    """
    try:
        from PIL import Image
        import io

        images = []
        for fig in figures:
            img_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            images.append(img)

        if images:
            images[0].save(
                str(path),
                save_all=True,
                append_images=images[1:] if len(images) > 1 else [],
                resolution=150
            )
            logger.info(f"Exported {len(figures)} figures to {path}")

    except ImportError:
        raise ExportError("Pillow not installed. Install with: pip install Pillow")
    except Exception as e:
        raise ExportError(f"Failed to export PDF: {e}")

