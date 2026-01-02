"""
PDF-to-Markdown pipeline using PyMuPDF for rendering and a pluggable LLM callback.

Workflow
--------
1) Convert each PDF page to an image (configurable DPI/format) in a temporary workspace.
2) Call an LLM (or any vision-to-text function) per page to obtain Markdown.
3) Persist per-page Markdown in temp; merge into a final Markdown file per PDF in `output_dir`.

Requirements
------------
pip install pymupdf pillow

Quickstart
----------
```python
from Python.pdf_to_markdown import process_pdfs_to_markdown

def my_llm(img_path: Path) -> str:
    # Replace with your real call (OpenAI Vision, local VLM, etc.)
    return f"# Page from {img_path.name}\\n\\n_Dummy content_"

final_md_files, per_page_md = process_pdfs_to_markdown(
    ["sample.pdf"],
    output_dir="out",
    page_image_dpi=200,
    image_format="PNG",
    llm_page_markdown_fn=my_llm,
)
print("Merged files:", final_md_files)
```
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image


__all__ = [
    "process_pdfs_to_markdown",
    "process_pdfs_to_markdown_with_tempdir",
    "convert_pdf_to_markdown",
]


def process_pdfs_to_markdown(
    pdf_paths: Iterable[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Tuple[list[Path], list[Path]]:
    """
    Convert PDFs to Markdown using an auto-cleaned temporary workspace.

    This is the safest default: intermediates live in a TemporaryDirectory that
    is removed automatically. Use `process_pdfs_to_markdown_with_tempdir` if you
    want to control the temp directory location and lifecycle.
    """

    with tempfile.TemporaryDirectory(prefix="pdf2img-md_") as temp_root:
        return _process_pdfs_to_markdown(
            pdf_paths=pdf_paths,
            output_dir=output_dir,
            temp_root=Path(temp_root),
            page_image_dpi=page_image_dpi,
            image_format=image_format,
            llm_page_markdown_fn=llm_page_markdown_fn,
        )


def process_pdfs_to_markdown_with_tempdir(
    pdf_paths: Iterable[str | os.PathLike],
    output_dir: str | os.PathLike,
    temp_root: str | os.PathLike,
    *,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Tuple[list[Path], list[Path]]:
    """
    Convert PDFs to Markdown using a user-managed temp directory.

    Choose this variant when you want to inspect or reuse the generated images
    and per-page Markdown. Cleanup is your responsibility.
    """

    temp_root_path = Path(temp_root)
    temp_root_path.mkdir(parents=True, exist_ok=True)

    return _process_pdfs_to_markdown(
        pdf_paths=pdf_paths,
        output_dir=output_dir,
        temp_root=temp_root_path,
        page_image_dpi=page_image_dpi,
        image_format=image_format,
        llm_page_markdown_fn=llm_page_markdown_fn,
    )


def convert_pdf_to_markdown(
    pdf_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Path:
    """
    Convenience wrapper for a single PDF -> Markdown conversion.
    """

    final_files, _ = process_pdfs_to_markdown(
        [pdf_path],
        output_dir=output_dir,
        page_image_dpi=page_image_dpi,
        image_format=image_format,
        llm_page_markdown_fn=llm_page_markdown_fn,
    )
    return final_files[0] if final_files else Path(output_dir) / f"{Path(pdf_path).stem}.md"


def _process_pdfs_to_markdown(
    pdf_paths: Iterable[str | os.PathLike],
    output_dir: str | os.PathLike,
    temp_root: Path,
    *,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Tuple[list[Path], list[Path]]:
    """
    Internal worker shared by the public entry points.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def llm_stub_markdown_from_image(img_path: Path) -> str:
        """
        Default placeholder for vision-to-Markdown; replace with your LLM/VLM call.
        """

        return (
            f"# Page extracted (stub)\n\n"
            f"_Image: {img_path.name}_\n\n"
            f"> Replace this with real LLM Markdown output."
        )

    llm_to_md = llm_page_markdown_fn or llm_stub_markdown_from_image

    final_markdown_files: list[Path] = []
    per_page_markdown_files_flattened: list[Path] = []

    for pdf_path in map(Path, pdf_paths):
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            # Skip invalid entries; adjust to raise if strict validation is desired.
            continue

        pdf_stem = pdf_path.stem
        pdf_temp_dir = temp_root / pdf_stem
        images_dir = pdf_temp_dir / "images"
        md_dir = pdf_temp_dir / "md"
        images_dir.mkdir(parents=True, exist_ok=True)
        md_dir.mkdir(parents=True, exist_ok=True)

        # --- 1) Render pages to images in temp ---
        pages_rendered: list[Path] = []
        with fitz.open(pdf_path) as doc:
            zoom = page_image_dpi / 72.0  # base DPI is ~72; convert to zoom factor
            mat = fitz.Matrix(zoom, zoom)

            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes(output=image_format.lower())

                img_name = f"page-{page_index + 1}.{image_format.lower()}"
                img_path = images_dir / img_name

                # Save via Pillow to standardize encoding/metadata.
                with Image.open(io.BytesIO(img_bytes)) as im:
                    im.save(img_path, format=image_format)

                pages_rendered.append(img_path)

        # --- 2) For each page image, call LLM to get Markdown; save per-page MD in temp ---
        page_md_files: list[Path] = []
        for img_path in pages_rendered:
            md_text = llm_to_md(img_path)
            md_path = md_dir / f"{img_path.stem}.md"
            md_path.write_text(md_text, encoding="utf-8")
            page_md_files.append(md_path)
            per_page_markdown_files_flattened.append(md_path)

        # --- 3) Merge per-page MD into FINAL non-temp Markdown file (one per PDF) ---
        final_md_path = output_dir / f"{pdf_stem}.md"
        with final_md_path.open("w", encoding="utf-8") as fout:
            fout.write(f"<!-- Source PDF: {pdf_path.name} -->\n")
            fout.write(f"# {pdf_stem}\n\n")
            for i, md_file in enumerate(sorted(page_md_files, key=lambda p: p.name), start=1):
                fout.write(f"\n\n---\n\n<!-- Page {i} -->\n\n")
                fout.write(md_file.read_text(encoding="utf-8"))

        final_markdown_files.append(final_md_path)

    return final_markdown_files, per_page_markdown_files_flattened
