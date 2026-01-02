"""
Reusable helpers for building Chroma vector stores with Hugging Face embeddings.

What this module does
---------------------
- Configures persistent Chroma collections for summaries and full documents.
- Adds documents (optionally chunked) while de-duplicating by ID.
- Runs similarity and MMR search, merging results with unique texts.
- Provides convenience cleanup for the persistence directory.

Dependencies
------------
pip install langchain-community langchain-huggingface sentence-transformers chromadb torch

Quickstart
----------
```python
from Python.chroma_utils import (
    setup_chroma_collections,
    add_documents,
    add_chunked_documents,
    query_collection,
)

summary_col, docs_col = setup_chroma_collections(
    persist_directory="./chroma_db",
    embedding_model_name="sentence-transformers/gtr-t5-base",
)

add_documents(
    summary_col,
    [
        {"id": "file1", "text": "Summary of Python's versatility."},
        {"id": "file2", "text": "Summary of ChromaDB as a vector database."},
    ],
)

add_chunked_documents(
    docs_col,
    [
        {"id": "file1", "chunk_id": 1, "text": "Python is a versatile programming language."},
        {"id": "file2", "chunk_id": 1, "text": "ChromaDB is a vector database for retrieval."},
    ],
)

results = query_collection(summary_col, "What is ChromaDB?", n_results=5)
for r in results:
    print(r.page_content, r.metadata)
```
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

__all__ = [
    "setup_chroma_collections",
    "add_documents",
    "add_chunked_documents",
    "query_collection",
    "delete_chroma_db",
    "embed_text",
]


def setup_chroma_collections(
    *,
    persist_directory: str | os.PathLike = "./chroma_db",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[Chroma, Chroma]:
    """
    Configure two Chroma collections (summary + docs) with persistent storage.

    Args:
        persist_directory: Directory for Chroma persistence.
        embedding_model_name: Hugging Face embedding model name.

    Returns:
        (summary_collection, docs_collection)
    """

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    persist_directory = str(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    summary_collection = Chroma(
        collection_name="literature_review_summary",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    docs_collection = Chroma(
        collection_name="literature_review_docs",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return summary_collection, docs_collection


def _existing_ids(collection: Chroma) -> set[str]:
    """Return the set of existing IDs in a Chroma collection."""

    data = collection.get()
    return set(data.get("ids", []))


def add_documents(collection: Chroma, documents: Iterable[dict]) -> None:
    """
    Add documents to a Chroma collection if not already present.

    Each document should provide:
        - id: unique identifier (used as the vector ID)
        - text: content to embed
    """

    existing = _existing_ids(collection)
    new_texts: List[str] = []
    new_ids: List[str] = []
    new_meta: List[dict] = []

    for doc in documents:
        doc_id = str(doc["id"])
        if doc_id in existing:
            continue
        new_ids.append(doc_id)
        new_texts.append(doc["text"])
        new_meta.append({"id": doc_id})

    if new_texts:
        collection.add_texts(texts=new_texts, metadatas=new_meta, ids=new_ids)


def add_chunked_documents(collection: Chroma, documents: Iterable[dict]) -> None:
    """
    Add chunked documents to a Chroma collection if not already present.

    Each document should provide:
        - id: base document identifier
        - chunk_id: integer or string chunk identifier
        - text: content to embed
    """

    existing = _existing_ids(collection)
    new_texts: List[str] = []
    new_ids: List[str] = []
    new_meta: List[dict] = []

    for doc in documents:
        base_id = str(doc["id"])
        chunk_id = str(doc.get("chunk_id", "0"))
        vector_id = f"{base_id}:{chunk_id}"
        if vector_id in existing:
            continue
        new_ids.append(vector_id)
        new_texts.append(doc["text"])
        new_meta.append({"id": base_id, "chunk_id": chunk_id})

    if new_texts:
        collection.add_texts(texts=new_texts, metadatas=new_meta, ids=new_ids)


def embed_text(text: str, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    Create an embedding for a single string using a Hugging Face model.
    """

    emb = HuggingFaceEmbeddings(model_name=model_name)
    return emb.embed_query(text)


def query_collection(
    collection: Chroma,
    query_text: str,
    *,
    n_results: int = 5,
) -> List[Document]:
    """
    Run similarity + MMR search and return unique Documents.
    """

    similarity_results = collection.similarity_search(query_text, k=n_results)
    mmr_results = collection.max_marginal_relevance_search(query_text, k=n_results)

    unique_by_text = {}
    for doc in similarity_results + mmr_results:
        unique_by_text.setdefault(doc.page_content, doc)
    return list(unique_by_text.values())


def delete_chroma_db(persist_directory: str | os.PathLike = "./chroma_db") -> None:
    """
    Delete the Chroma persistence directory.
    """

    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted Chroma database at {path}")
    else:
        print(f"Chroma database not found at {path}")
