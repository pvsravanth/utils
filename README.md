# UTILS

Reusable utilities and code snippets collected by category.

## Python

- **Paddle OCR bounding boxes** (`Python/paddle_ocr_bounding_boxes.py`): Wrapper around `paddleocr` to detect text, draw bounding boxes, and save annotated images. Supports paths, Pillow images, or numpy arrays with configurable colors, label backgrounds, and confidence filtering.
- **Decorators** (`Python/decorators.py`): Common decorators for logging, timing, retries with backoff, rate limiting, caching with TTL, deprecation warnings, and teardown hooks.
- **Agent logger** (`Python/agent_logger.py`): Structured JSON logging for agents with async/sync decorators, trace IDs, cost estimation from token usage, and safe output previews.
- **PDF → Markdown pipeline** (`Python/pdf_to_markdown.py`): Render PDFs to page images, run a pluggable LLM per page to produce Markdown, and merge into one Markdown file per PDF. Includes auto-cleaned temp flow and a variant that uses a provided temp directory.
- **DocLayout processor** (`Python/doclayout_processor.py`): Convert PDFs to images and run YOLOv10 DocLayout to annotate pages with class-colored bounding boxes.
- **Audio channel activity** (`Python/audio_channel_activity.py`): Detect which stereo channel is more active over time to approximate speaker activity.
- **Chroma utilities** (`Python/chroma_utils.py`): Set up persistent Chroma collections with Hugging Face embeddings, add (chunked) documents, query via similarity + MMR, and clean up storage.

### Setup

```bash
pip install "paddleocr>=2.7" pillow numpy pymupdf pdf2image opencv-python doclayout-yolo langchain-community langchain-huggingface sentence-transformers chromadb torch
```

### PaddleOCR quickstart

```python
from Python.paddle_ocr_bounding_boxes import PaddleOCRBoundingBoxDrawer

drawer = PaddleOCRBoundingBoxDrawer(lang="en")
path = drawer.annotate_image("sample.jpg", box_color="lime", text_color="black")
print(f"Saved: {path}")
```

### Decorators quickstart

```python
from Python.decorators import logged, retry, cached

@logged(log_args=True, log_result=True)
def add(a, b):
    return a + b

@retry(attempts=5, base_delay_s=0.1)
def flaky_call():
    ...

@cached(ttl_s=60)
def get_config():
    ...
```

### Agent logger quickstart

```python
from Python.agent_logger import log_agent

@log_agent("demo-agent")
async def run_task():
    return {"message": "hello"}
```

### PDF → Markdown quickstart

```python
from Python.pdf_to_markdown import (
    process_pdfs_to_markdown,
    process_pdfs_to_markdown_with_tempdir,
    convert_pdf_to_markdown,
)

def my_llm(img_path):
    return f"# Page from {img_path.name}\\n\\n_Dummy content_"

# Auto-cleaned temp directory (best default)
final_md_files, per_page_md = process_pdfs_to_markdown(
    ["sample.pdf"],
    output_dir="out",
    page_image_dpi=200,
    image_format="PNG",
    llm_page_markdown_fn=my_llm,
)
print(final_md_files)

# Explicit temp directory you manage/inspect
process_pdfs_to_markdown_with_tempdir(
    ["sample.pdf"],
    output_dir="out",
    temp_root="/tmp/pdf-to-md",
    llm_page_markdown_fn=my_llm,
)

# Convenience wrapper for a single PDF
convert_pdf_to_markdown("sample.pdf", output_dir="out", llm_page_markdown_fn=my_llm)
```

### DocLayout processor quickstart

```python
from Python.doclayout_processor import DocumentLayoutProcessor

processor = DocumentLayoutProcessor(
    pdf_path="sample.pdf",
    output_dir="out/pages",
    model_weights="weights/doclayout-yolov10.pt",
    image_format="jpg",
)
processor.convert_pdf_to_images()
processor.process_images(confidence=0.25)
```

### Audio channel activity quickstart

```python
from Python.audio_channel_activity import get_channel_activity

activity = get_channel_activity("call.wav", frame_duration_ms=500, noise_threshold=5000)
for ts, speaker in activity:
    print(f"{ts:.2f}s -> {speaker}")
```

### Chroma utilities quickstart

```python
from Python.chroma_utils import (
    setup_chroma_collections,
    add_documents,
    add_chunked_documents,
    query_collection,
)

summary_col, docs_col = setup_chroma_collections(persist_directory="./chroma_db")
add_documents(summary_col, [{"id": "file1", "text": "Summary of ChromaDB."}])
add_chunked_documents(docs_col, [{"id": "file1", "chunk_id": 1, "text": "ChromaDB is a vector database."}])
results = query_collection(summary_col, "What is ChromaDB?", n_results=5)
```

## Notes

- PaddleOCR downloads models on first use; ensure network access or pre-download in your environment.
- Adjust logging levels in `Python/decorators.py` via the module logger (`decorators`).
- `Python/agent_logger.py` uses `python-json-logger` if available; it falls back to a built-in JSON formatter otherwise.
- `Python/pdf_to_markdown.py` requires `pymupdf` and `pillow`; swap the stub LLM with your own vision-to-Markdown call.
- `Python/doclayout_processor.py` requires `pdf2image`, `opencv-python`, and `doclayout-yolo`; pdf2image may need poppler installed.
- `Python/audio_channel_activity.py` requires numpy and expects stereo WAV input.
- `Python/chroma_utils.py` relies on langchain-community, langchain-huggingface, sentence-transformers, chromadb, and torch.
