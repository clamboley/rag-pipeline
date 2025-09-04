"""Offline RAG Pipeline for Code Documentation Retrieval."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.utils import read_file, search_files, setup_logger

DEFAULT_EXTENSIONS = [".md", ".txt", ".rst", ".py"]
logger = setup_logger(__name__)


@dataclass
class Document:
    """Document chunk with metadata."""

    idx: str
    text: str
    source: str
    start_char: int
    end_char: int


class OfflineCodeDocRAG:
    """Offline RAG pipeline for code documentation retrieval."""

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str = "./rag_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the offline RAG pipeline.

        Args:
            model_path (str): Path to local HuggingFace model directory.
            index_path (str): Directory to store the FAISS index and metadata.
            chunk_size (int): Size of text chunks in characters.
            chunk_overlap (int): Overlap between chunks in characters.
            max_length (int, optional): Maximum input length for embedding model (None for max).
            device (str, optional): Device for model inference ('cuda', 'cpu', or None for auto).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)

        self.device = device
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model from local directory: {model_path}")
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        self.max_length = max_length or self.tokenizer.model_max_length
        self.embedding_dim = self.model.config.hidden_size
        self.index_file = self.index_path / "faiss.index"
        self.metadata_file = self.index_path / "metadata.json"

        if self.index_file.exists() and self.metadata_file.exists():
            self._load_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new FAISS index and metadata store."""
        # Using IndexFlatIP for inner product
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")

    def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        self.index = faiss.read_index(str(self.index_file))

        with Path.open(self.metadata_file, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)

        self.documents = [Document(**doc) for doc in raw_documents]
        logger.info(f"Loaded existing index with {len(self.documents)} documents")

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_file))

        raw_docs = [vars(doc) for doc in self.documents]
        with Path.open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(raw_docs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved index with {len(self.documents)} documents")

    def mean_pooling(self, model_output: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings."""
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        mask = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Encode texts to embeddings using the local model.

        Args:
            texts (list[str]): list of texts to encode.
            batch_size (int): Number of embeddings to generate at once.

        Returns:
            Numpy array of embeddings.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            token_ids = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Generate embeddings
            model_output = self.model(**token_ids)
            embeddings = self.mean_pooling(model_output, token_ids["attention_mask"])

            # Normalize embeddings for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def chunk_text(self, text: str, source: str = "") -> list[Document]:
        """Split text into overlapping token chunks, preserving char offsets."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        tokens = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        total_tokens = len(tokens)

        chunks = []
        start_token = 0

        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_size, total_tokens)

            # Character positions from first and last token
            start_char = offsets[start_token][0]
            end_char = offsets[end_token - 1][1]

            chunk_text = text[start_char:end_char].strip()

            if chunk_text:
                chunk_id = f"{source}:{start_char}-{end_char}"
                chunks.append(
                    Document(
                        idx=hashlib.sha256(chunk_id.encode()).hexdigest(),
                        text=chunk_text,
                        source=source,
                        start_char=start_char,
                        end_char=end_char,
                    ),
                )

            start_token += self.chunk_size - self.chunk_overlap

        return chunks

    def add_documents(
        self,
        documents: list[dict[str, str]],
        batch_size: int = 32,
        *,
        save_after_adding: bool = True,
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents (list[dict[str, str]]): list of dicts with 'content' and optional 'source'.
            batch_size: Number of embeddings to generate at once.
            save_after_adding (bool): Whether to save the index after adding documents.
        """
        all_chunks = []

        logger.info("Chunking documents...")
        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            chunks = self.chunk_text(content, source)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")

        logger.info("Generating embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.encode_texts(texts, batch_size=batch_size)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store documents metadata
        for chunk in all_chunks:
            self.documents.append(chunk)

        logger.info(f"Successfully indexed {len(all_chunks)} chunks.")

        if save_after_adding:
            self.save_index()

    def add_files(
        self,
        file_paths: list[str | Path],
        extensions: list[str] = DEFAULT_EXTENSIONS,
        *,
        save_after_adding: bool = True,
    ) -> None:
        """Add documentation files to the vector database.

        Args:
            file_paths (list[str | Path]): List of file paths or directory paths
            extensions (list[str]): List of file extensions to search for.
            save_after_adding (bool): Whether to save the index after adding files.
        """
        documents = []

        logger.info(f"{file_paths = }")

        for path in file_paths:
            path_obj = Path(path)

            if path_obj.is_file() and any(path_obj.suffix == ext for ext in extensions):
                content_dict = read_file(path_obj)
                if content_dict:
                    documents.append(content_dict)

            elif path_obj.is_dir():
                # Directory - recursively find matching files
                for file_path in search_files(path_obj, extensions=extensions):
                    content_dict = read_file(file_path)
                    if content_dict:
                        documents.append(content_dict)

        if documents:
            logger.info(f"Found {len(documents)} files to index.")
            self.add_documents(documents, save_after_adding=save_after_adding)
        else:
            logger.info("No matching files found.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_source: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant documentation chunks for a query.

        Args:
            query (str): Search query.
            top_k (int): Number of results to return.
            filter_source (str, optional): Source file to filter results.

        Returns:
            List of dictionaries containing retrieved chunks and metadata.
        """
        if len(self.documents) == 0:
            logger.info("No documents in index. Please add documents first.")
            return []

        # Generate query embedding and search in FAISS
        query_embedding = self.encode_texts([query])[0].reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.documents)))

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            if filter_source and doc.source != filter_source:
                continue

            results.append(
                {
                    "text": doc.text,
                    "source": doc.source,
                    "score": float(score),
                    "metadata": {
                        "start_char": doc.start_char,
                        "end_char": doc.end_char,
                        "idx": doc.idx,
                    },
                },
            )

            if len(results) >= top_k:
                break

        return results

    def clear_index(self) -> None:
        """Clear the index and all stored documents."""
        self.create_new_index()

        if self.index_file.exists():
            self.index_file.unlink()

        if self.metadata_file.exists():
            self.metadata_file.unlink()

        logger.info("Index cleared")

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        return {
            "total_documents": len(self.documents),
            "index_size_bytes": self.index_file.stat().st_size if self.index_file.exists() else 0,
            "unique_sources": len({doc.source for doc in self.documents}),
            "embedding_dimension": self.embedding_dim,
        }
