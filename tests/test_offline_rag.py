"""Pytest suite for the OfflineCodeDocRAG pipeline."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.tokenization_utils_base import BatchEncoding

from src.offline_rag import OfflineCodeDocRAG
from src.splitting import make_splitter_for_file


@pytest.fixture
def dummy_model_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock HuggingFace AutoModel and AutoTokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 16
    mock_tokenizer.return_value = BatchEncoding(
        data={
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones((2, 16)),
        },
    )
    mock_tokenizer.to.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.hidden_size = 8
    mock_model.return_value = (torch.randn(2, 16, 8),)
    mock_model.to.return_value = mock_model

    monkeypatch.setattr(
        "src.offline_rag.AutoTokenizer.from_pretrained",
        lambda *_, **__: mock_tokenizer,
    )
    monkeypatch.setattr(
        "src.offline_rag.AutoModel.from_pretrained",
        lambda *_, **__: mock_model,
    )


@pytest.fixture
def rag(tmp_path: Path, dummy_model_and_tokenizer: None) -> OfflineCodeDocRAG:
    """Fixture for a fresh OfflineCodeDocRAG instance."""
    return OfflineCodeDocRAG(
        model_path=tmp_path,
        index_path=tmp_path / "index",
        chunk_size=50,
        chunk_overlap=10,
        index_type="flat",
    )


# ---------------------------------------------------------------------------
# Tests for splitting.py
# ---------------------------------------------------------------------------


def test_make_splitter_for_known_extension(tmp_path: Path) -> None:
    """Ensure splitter is created with language support for known extensions."""
    splitter = make_splitter_for_file(tmp_path.with_suffix(".py"), 100, 20)
    assert isinstance(splitter, RecursiveCharacterTextSplitter), "Wrong splitter type."
    assert splitter._chunk_size == 100, "Wrong chunk size."
    assert splitter._chunk_overlap == 20, "Wrong chunk overlap."


def test_make_splitter_for_unknown_extension(tmp_path: Path) -> None:
    """Fallback splitter should work when extension is not recognized."""
    splitter = make_splitter_for_file(tmp_path.with_suffix(".unknown"), 200, 40)
    assert isinstance(splitter, RecursiveCharacterTextSplitter), "Wrong splitter type."
    assert splitter._chunk_size == 200, "Wrong chunk size."
    assert splitter._chunk_overlap == 40, "Wrong chunk overlap."


# ---------------------------------------------------------------------------
# Tests for OfflineCodeDocRAG
# ---------------------------------------------------------------------------


def test_chunk_text_creates_documents(rag: OfflineCodeDocRAG) -> None:
    """Chunking should produce Document objects with metadatas."""
    docs = rag.chunk_text("def foo():\n    return 42", source="example.py")
    assert all(isinstance(doc, Document) for doc in docs), "Not all elements are Document objects."
    assert all("id" in doc.metadata for doc in docs), "Missing id in metadata."
    assert all("source" in doc.metadata for doc in docs), "Missing source in metadata."
    assert all("start_index" in doc.metadata for doc in docs), "Missing start_index in metadata."


def test_encode_texts_returns_numpy_array(rag: OfflineCodeDocRAG) -> None:
    """Encoding texts should return a numpy array with correct shape."""
    embeddings = rag.encode_texts(["hello world", "another text"])
    assert isinstance(embeddings, np.ndarray), "Embeddings should be a numpy array."
    assert embeddings.shape[1] == rag.embedding_dim, "Embedding dimension mismatch."


def test_add_documents_adds_and_indexes(rag: OfflineCodeDocRAG) -> None:
    """Adding documents should increase stored documents and index size."""
    docs = [{"content": "print('hello')", "source": "file1.py"}]
    rag.add_documents(docs, save_after_adding=False)
    stats = rag.get_stats()
    assert stats["total_documents"] > 0, "No documents were added."


def test_retrieve_returns_results(rag: OfflineCodeDocRAG) -> None:
    """Retrieve should return relevant chunks for a query."""
    rag.add_documents(
        [
            {"content": "def add(a, b): return a + b", "source": "math.py"},
            {"content": "def sub(a, b): return a - b", "source": "math.py"},
        ],
        save_after_adding=False,
    )
    results = rag.retrieve("add function", top_k=2)
    assert isinstance(results, list), "Results should be a list."
    assert len(results) > 0, "No results were returned."
    assert all("text" in r for r in results), "Results should contain 'text' field."
    assert all("metadata" in r for r in results), "Results should contain 'metadata' field."


def test_clear_index_resets_documents(rag: OfflineCodeDocRAG) -> None:
    """Clearing the index should reset stored documents."""
    rag.add_documents([{"content": "sample text", "source": "test.md"}], save_after_adding=False)
    assert rag.get_stats()["total_documents"] > 0, "Documents were not added."
    rag.clear_index()
    assert rag.get_stats()["total_documents"] == 0, "Index was not cleared."


def test_invalid_weights_raise_error(rag: OfflineCodeDocRAG) -> None:
    """Retrieve should raise ValueError when both weights are <= 0."""
    with pytest.raises(ValueError, match="Invalid weights:"):
        rag.retrieve("query", semantic_weight=0.0, bm25_weight=0.0)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_add_files_with_single_file(rag: OfflineCodeDocRAG, tmp_path: Path) -> None:
    """Integration: add_files should index a single .py file end-to-end."""
    file_path = tmp_path / "hello.py"
    file_path.write_text("def greet():\n    return 'hello'\n")

    rag.add_files([file_path], save_after_adding=False)
    stats = rag.get_stats()
    assert stats["total_documents"] > 0, "No documents were indexed."
    results = rag.retrieve("greet", top_k=1)
    assert any("greet" in r["text"] for r in results), "Results do not contain the expected text."


def test_add_files_with_directory(rag: OfflineCodeDocRAG, tmp_path: Path) -> None:
    """Integration: add_files should find and index files from a directory."""
    dir_path = tmp_path / "src"
    dir_path.mkdir()

    mardown_text = textwrap.dedent("""\
        # Title
        Some markdown docs.
    """)
    (dir_path / "b.md").write_text(mardown_text)
    (dir_path / "a.py").write_text("def alpha():\n    return 1\n")

    rag.add_files([dir_path], save_after_adding=False)
    stats = rag.get_stats()
    assert stats["total_documents"] >= 2, "Expected at least 2 documents to be indexed."

    results = rag.retrieve("alpha", top_k=2)
    assert any("alpha" in r["text"] for r in results), "Results do not contain the expected text."
    assert any("Title" in r["text"] for r in results), "Results do not contain the expected text."
