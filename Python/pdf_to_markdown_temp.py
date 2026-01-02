"""
Temp-directory–oriented helpers for PDF → Markdown conversion.

This module exposes only the temp-managed variant so you can choose exactly
where intermediates (page images, per-page Markdown) are written and when they
are cleaned up. It delegates to `process_pdfs_to_markdown_with_tempdir` in
`pdf_to_markdown.py`.

Quickstart
----------
```python
from Python.pdf_to_markdown_temp import convert_pdfs_with_tempdir

def my_llm(img_path):
    return f"# Page from {img_path.name}\\n\\n_Dummy content_"

final_md_files, per_page_md = convert_pdfs_with_tempdir(
    ["sample.pdf"],
    output_dir="out",
    temp_root="/tmp/pdf-md",
    page_image_dpi=200,
    image_format="PNG",
    llm_page_markdown_fn=my_llm,
)
print("Merged:", final_md_files)
print("Per-page (temp):", per_page_md)
```

Tempfile patterns cheat sheet
-----------------------------
```python
import tempfile
from pathlib import Path

# Auto-cleaned directory (recommended)
with tempfile.TemporaryDirectory(prefix="myjob_") as tmp:
    tmp_path = Path(tmp)
    (tmp_path / "example.txt").write_text("hello tempdir")

# User-managed directory you clean up later
temp_root = Path(tempfile.mkdtemp(prefix="myjob_"))
try:
    (temp_root / "file.bin").write_bytes(b"data")
finally:
    import shutil
    shutil.rmtree(temp_root, ignore_errors=True)

# Auto-cleaned file object with a real name on disk
with tempfile.NamedTemporaryFile(prefix="note_", suffix=".txt", delete=True) as f:
    f.write(b"hi")
    f.flush()
    print("temp file path:", f.name)

# Low-level fd + path (you must close + remove)
fd, path = tempfile.mkstemp(prefix="raw_", suffix=".dat")
try:
    with os.fdopen(fd, "wb") as fh:
        fh.write(b"bytes")
finally:
    Path(path).unlink(missing_ok=True)

# In-memory until size threshold, then spills to disk
with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as spooled:
    spooled.write(b"small payload")
    spooled.seek(0)
    print(spooled.read())
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

# Support both package-style and script-style imports.
try:  # pragma: no cover - import convenience
    from .pdf_to_markdown import process_pdfs_to_markdown_with_tempdir  # type: ignore
except ImportError:  # pragma: no cover - fallback when not used as a package
    from pdf_to_markdown import process_pdfs_to_markdown_with_tempdir  # type: ignore


__all__ = ["convert_pdfs_with_tempdir", "convert_pdf_with_tempdir"]


def convert_pdfs_with_tempdir(
    pdf_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    temp_root: str | Path,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Tuple[list[Path], list[Path]]:
    """
    Convert multiple PDFs to Markdown using a user-managed temp directory.

    Args:
        pdf_paths: Iterable of PDF paths to process.
        output_dir: Directory where merged Markdown files are written.
        temp_root: Directory for all intermediates (images, per-page Markdown).
        page_image_dpi: Rendering DPI for page images.
        image_format: Output image format (e.g., "PNG", "JPEG").
        llm_page_markdown_fn: Callable that maps a page image path to Markdown.

    Returns:
        (final_markdown_files, per_page_markdown_files_flattened)
    """

    return process_pdfs_to_markdown_with_tempdir(
        pdf_paths=pdf_paths,
        output_dir=output_dir,
        temp_root=temp_root,
        page_image_dpi=page_image_dpi,
        image_format=image_format,
        llm_page_markdown_fn=llm_page_markdown_fn,
    )


def convert_pdf_with_tempdir(
    pdf_path: str | Path,
    *,
    output_dir: str | Path,
    temp_root: str | Path,
    page_image_dpi: int = 200,
    image_format: str = "PNG",
    llm_page_markdown_fn: Optional[Callable[[Path], str]] = None,
) -> Path:
    """
    Convenience wrapper for a single PDF using a user-managed temp directory.
    """

    final_files, _ = convert_pdfs_with_tempdir(
        [pdf_path],
        output_dir=output_dir,
        temp_root=temp_root,
        page_image_dpi=page_image_dpi,
        image_format=image_format,
        llm_page_markdown_fn=llm_page_markdown_fn,
    )
    return final_files[0] if final_files else Path(output_dir) / f"{Path(pdf_path).stem}.md"
