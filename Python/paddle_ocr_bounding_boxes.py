"""
Utilities for running PaddleOCR and drawing bounding boxes on images.

This module wraps the `paddleocr` package in a small, well-documented class that:
1. Runs OCR on an input image (file path, Pillow image, or ndarray).
2. Draws each detected bounding box with the recognized text and confidence score.
3. Returns a Pillow image or saves an annotated file for downstream workflows.

Usage example
-------------
```python
from paddle_ocr_bounding_boxes import PaddleOCRBoundingBoxDrawer

drawer = PaddleOCRBoundingBoxDrawer(lang="en")
annotated_path = drawer.annotate_image(
    image_path="sample.jpg",
    output_path="sample_annotated.jpg",
    box_color="lime",
    text_color="black",
)
print(f"Annotated image saved to: {annotated_path}")

# Or work fully in-memory with Pillow:
from PIL import Image

image = Image.open("sample.jpg")
detections = drawer.run_ocr(image, score_threshold=0.5)
annotated_image = drawer.draw_detections(
    image=image,
    detections=detections,
    box_color="cyan",
    label_background="white",
)
annotated_image.save("sample_annotated.jpg")
```

Installation note
-----------------
The first call to PaddleOCR downloads detection/recognition models. Ensure the
runtime has network access and run:

    pip install "paddleocr>=2.7" pillow numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR


# A bounding box is a list of four points (x, y) defining a quadrilateral polygon.
Point = Tuple[float, float]
Box = Sequence[Point]
ImageInput = Path | str | Image.Image | np.ndarray

__all__ = [
    "OCRDetection",
    "PaddleOCRBoundingBoxDrawer",
]


@dataclass
class OCRDetection:
    """
    Simple DTO capturing the detection result for one text region.

    Attributes:
        box: The quadrilateral polygon enclosing the detected text.
        text: The recognized text string.
        confidence: Probability score provided by PaddleOCR.
    """

    box: Box
    text: str
    confidence: float


class PaddleOCRBoundingBoxDrawer:
    """
    Run PaddleOCR and draw the detected bounding boxes on an image.

    The class encapsulates model initialization and visualization so that callers
    can focus on providing an image path and receiving an annotated output. The
    constructor arguments are passed directly to `PaddleOCR` for flexibility.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, **ocr_kwargs):
        """
        Initialize the PaddleOCR engine.

        Args:
            lang: Language code passed to PaddleOCR (for example, "en" or "ch").
            use_angle_cls: Whether to enable angle classification for rotated text.
            **ocr_kwargs: Any additional keyword args supported by `PaddleOCR`.
        """

        # PaddleOCR lazily downloads models on first run; keep initialization simple.
        self.ocr = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, **ocr_kwargs)

    def run_ocr(self, image: ImageInput, score_threshold: float = 0.0) -> List[OCRDetection]:
        """
        Execute OCR on the provided image input and return structured detections.

        Args:
            image: File path, Pillow image, or numpy ndarray.
            score_threshold: Minimum confidence score (0.0-1.0) to keep a detection.

        Returns:
            List of OCRDetection items, one per detected text region.
        """

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")

        prepared_image = self._prepare_image_input(image)
        results = self.ocr.ocr(prepared_image, cls=True)

        detections: List[OCRDetection] = []
        for line in results:
            for box, (text, confidence) in line:
                if confidence >= score_threshold:
                    detections.append(
                        OCRDetection(box=box, text=text, confidence=confidence)
                    )
        return detections

    def annotate_image(
        self,
        image_path: Path | str,
        output_path: Path | str | None = None,
        box_color: str = "red",
        box_width: int = 3,
        text_color: str = "white",
        font: ImageFont.ImageFont | None = None,
        label_background: str | None = "black",
        show_confidence: bool = True,
        score_threshold: float = 0.0,
        padding: int = 2,
    ) -> Path:
        """
        Draw bounding boxes and recognized text on the image.

        Args:
            image_path: Source image path.
            output_path: Optional output path; defaults to `<name>_annotated<ext>`.
            box_color: Outline color for bounding boxes (any Pillow-compatible color).
            box_width: Outline thickness in pixels.
            text_color: Color for text labels.
            font: Optional PIL ImageFont; defaults to `ImageFont.load_default()`.
            label_background: Optional fill color behind text labels; set to None for no background.
            show_confidence: Whether to append confidence scores to the label text.
            score_threshold: Minimum confidence score (0.0-1.0) to keep a detection.
            padding: Extra pixels of padding around label backgrounds.

        Returns:
            Path to the saved annotated image.
        """

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        annotated_path = self._resolve_output_path(image_path, output_path)
        font = font or ImageFont.load_default()

        detections = self.run_ocr(image_path, score_threshold=score_threshold)
        base_image = Image.open(image_path).convert("RGB")
        annotated_image = self.draw_detections(
            image=base_image,
            detections=detections,
            box_color=box_color,
            box_width=box_width,
            text_color=text_color,
            label_background=label_background,
            show_confidence=show_confidence,
            font=font,
            padding=padding,
        )

        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(annotated_path)
        return annotated_path

    def draw_detections(
        self,
        image: Image.Image,
        detections: Sequence[OCRDetection],
        box_color: str = "red",
        box_width: int = 3,
        text_color: str = "white",
        label_background: str | None = "black",
        show_confidence: bool = True,
        font: ImageFont.ImageFont | None = None,
        padding: int = 2,
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on an in-memory PIL image.

        Args:
            image: Input image (will not be modified; a copy is returned).
            detections: OCR detections to render.
            box_color: Outline color for bounding boxes.
            box_width: Outline thickness in pixels.
            text_color: Color for text labels.
            label_background: Optional fill color behind text labels; set to None for no background.
            show_confidence: Whether to append confidence scores to the label text.
            font: Optional PIL ImageFont; defaults to `ImageFont.load_default()`.
            padding: Extra pixels of padding around label backgrounds.

        Returns:
            A new `Image.Image` instance containing the drawn annotations.
        """

        font = font or ImageFont.load_default()
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        for detection in detections:
            polygon = [(float(x), float(y)) for x, y in detection.box]
            draw.polygon(polygon, outline=box_color, width=box_width)

            # Position label near the top-left of the polygon to avoid overlap.
            label = (
                detection.text
                if not show_confidence
                else f"{detection.text} ({detection.confidence:.2f})"
            )
            label_position = self._label_position(polygon, padding=padding)
            self._draw_label(
                draw=draw,
                label=label,
                position=label_position,
                font=font,
                text_color=text_color,
                background_color=label_background,
                padding=padding,
            )

        return annotated

    @staticmethod
    def _resolve_output_path(image_path: Path, output_path: Path | str | None) -> Path:
        """
        Generate an output path if one was not provided.

        The default follows `<stem>_annotated<suffix>` in the same directory.
        """

        if output_path is not None:
            return Path(output_path)
        return image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")

    @staticmethod
    def _label_position(polygon: Iterable[Point], padding: int = 2) -> Tuple[float, float]:
        """
        Choose a label anchor point (top-left corner of the polygon).
        """

        xs, ys = zip(*polygon)
        return min(xs) + padding, min(ys) + padding

    @staticmethod
    def _draw_label(
        draw: ImageDraw.ImageDraw,
        label: str,
        position: Tuple[float, float],
        font: ImageFont.ImageFont,
        text_color: str,
        background_color: str | None,
        padding: int = 2,
    ) -> None:
        """
        Render a label with optional background padding onto the image.
        """

        if not label:
            return

        text_bbox = draw.textbbox(position, label, font=font)
        x0, y0, x1, y1 = text_bbox

        if background_color is not None:
            draw.rectangle(
                (x0 - padding, y0 - padding, x1 + padding, y1 + padding),
                fill=background_color,
            )

        draw.text(position, label, fill=text_color, font=font)

    @staticmethod
    def _prepare_image_input(image: ImageInput) -> str | np.ndarray:
        """
        Normalize supported input types to what PaddleOCR expects (path or ndarray).
        """

        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            return str(image_path)

        raise TypeError(
            "Unsupported image input type. Provide a path, PIL.Image.Image, or numpy.ndarray."
        )
