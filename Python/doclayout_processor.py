"""
Reusable utilities for PDF-to-image conversion and document layout detection with YOLOv10 DocLayout.

Features
--------
- Convert PDFs to page images (skips conversion if images already exist).
- Run YOLOv10 DocLayout to detect layout elements per page.
- Draw class-colored bounding boxes and overwrite or save annotated images.

Dependencies
------------
pip install pdf2image pillow opencv-python doclayout-yolo
# pdf2image may require poppler on your system; see pdf2image docs.

Quickstart
----------
```python
from Python.doclayout_processor import DocumentLayoutProcessor

processor = DocumentLayoutProcessor(
    pdf_path="sample.pdf",
    output_dir="out/pages",
    model_weights="weights/doclayout-yolov10.pt",
    image_format="jpg",
)
processor.convert_pdf_to_images()          # renders pages if not already present
processor.process_images(confidence=0.25)  # detects layout and annotates images in-place
```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_path
from PIL import Image

Color = Tuple[int, int, int]

__all__ = ["DocumentLayoutProcessor", "DEFAULT_COLOR_MAP"]

logger = logging.getLogger(__name__)

# Default class-to-color mapping (BGR for OpenCV).
DEFAULT_COLOR_MAP: Dict[str, Color] = {
    "Title": (255, 0, 0),
    "Text": (0, 255, 0),
    "Section-header": (0, 0, 255),
    "List-item": (255, 255, 0),
    "Table": (255, 0, 255),
    "Figure": (0, 255, 255),
    "Formula": (128, 0, 128),
    "Footnote": (0, 128, 128),
    "Page-header/Footer": (128, 128, 0),
    "__default__": (100, 100, 100),
}


class DocumentLayoutProcessor:
    """
    Convert PDFs to images, run YOLOv10 DocLayout, and annotate pages with class-colored boxes.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        model_weights: str | Path,
        *,
        image_format: str = "jpg",
        color_map: Optional[Dict[str, Color]] = None,
    ) -> None:
        """
        Args:
            pdf_path: Input PDF path.
            output_dir: Directory to store page images (and annotated replacements).
            model_weights: Path to YOLOv10 DocLayout weights.
            image_format: Image format extension for saved pages (e.g., "jpg", "png").
            color_map: Optional mapping of class name -> BGR color tuple.
        """

        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.model_weights = Path(model_weights)
        self.image_format = image_format
        self.color_map = {**DEFAULT_COLOR_MAP, **(color_map or {})}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLOv10(str(self.model_weights))
        logger.info("Initialized DocumentLayoutProcessor for %s", self.pdf_path)

    def convert_pdf_to_images(self, *, dpi: int = 300, force: bool = False) -> None:
        """
        Render PDF pages to images if needed.

        Args:
            dpi: Render DPI for pdf2image; higher is sharper.
            force: If True, re-render even if images already exist.
        """

        if not force and self._images_exist():
            logger.info("Images already present in %s; skipping conversion.", self.output_dir)
            return

        logger.info("Converting PDF to images: %s", self.pdf_path)
        images = convert_from_path(self.pdf_path, dpi=dpi, fmt=self.image_format)
        for i, img in enumerate(images, start=1):
            img_path = self.output_dir / f"page_{i}.{self.image_format}"
            img.save(img_path)
            logger.debug("Saved image: %s", img_path)
        logger.info("PDF conversion complete; %d pages rendered.", len(images))

    def process_images(
        self,
        *,
        imgsz: int = 1024,
        confidence: float = 0.2,
        overwrite: bool = True,
    ) -> None:
        """
        Detect layout elements on each page image and draw bounding boxes.

        Args:
            imgsz: YOLO input image size (short side).
            confidence: Detection confidence threshold.
            overwrite: If True, overwrite images in-place with annotations.
        """

        logger.info("Processing images in %s", self.output_dir)

        for image_path in self._iter_images():
            results = self.model.predict(str(image_path), imgsz=imgsz, conf=confidence)
            if not results:
                logger.warning("No results returned for %s", image_path)
                continue
            result = results[0]

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read image: %s", image_path)
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names.get(class_id, str(class_id))
                color = self.color_map.get(class_name, self.color_map["__default__"])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    class_name,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            target_path = image_path if overwrite else image_path.with_name(f"{image_path.stem}_annotated.{self.image_format}")
            cv2.imwrite(str(target_path), image)
            logger.info("Annotated image saved: %s", target_path)

    def _iter_images(self) -> Iterable[Path]:
        """Yield image paths in output_dir matching the configured format."""

        for file in sorted(self.output_dir.iterdir()):
            if file.is_file() and file.suffix.lower().lstrip(".") == self.image_format.lower().lstrip("."):
                yield file

    def _images_exist(self) -> bool:
        """Check if any rendered images already exist."""

        return any(self._iter_images())
