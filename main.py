"""Offline RAG Pipeline for Code Documentation Retrieval."""

import argparse
from pathlib import Path

from src.offline_rag import OfflineCodeDocRAG
from src.utils import setup_logger

logger = setup_logger(__name__)

EXAMPLE_DOCUMENTS = [
    {
        "content": """
            def calculate_similarity(vec1, vec2):
                '''Calculate cosine similarity between two vectors.

                Args:
                    vec1: First vector as numpy array
                    vec2: Second vector as numpy array

                Returns:
                    Float value between -1 and 1 representing similarity
                '''
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                return dot_product / (norm1 * norm2)
            """,
        "source": "utils/similarity.py",
    },
    {
        "content": """
            ## Vector Database Operations

            The vector database supports the following operations:
            - Insert: Add new vectors with metadata
            - Search: Find k-nearest neighbors
            - Update: Modify existing vectors
            - Delete: Remove vectors by ID

            All operations are performed locally without network calls.
            """,
        "source": "docs/vector_db.md",
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        type=Path,
        nargs="+",
        help="List of files and directories to index.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default="rag_index",
        help="Directory to store the FAISS index and metadata",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default="models/Qwen3-Embedding-0.6B",
        help="Path to the local embedding model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Size of each document chunk in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks in characters.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="String that needs to be present in the source file path to be retrieved.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of document to retrieve from vector DB.",
    )
    parser.add_argument(
        "--filter-source",
        type=str,
        default=None,
        help="String that needs to be present in the source file path to be retrieved.",
    )
    args = parser.parse_args()

    rag = OfflineCodeDocRAG(
        model_path=args.model_path,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_type=args.index_type,
        device="cpu",
    )

    # Ex 1 : Add individual documents
    if not args.data_files:
        rag.add_documents(EXAMPLE_DOCUMENTS)

    # Ex 2: Add files from directory
    if args.data_files:
        rag.add_files(args.data_files)

    # Retrieve relevant documentation
    query = "How to chunk a text in a RAG pipeline?"
    results = rag.retrieve(query, top_k=args.top_k, filter_source=args.filter_source)

    print(f'Query: "{query}"')
    print(f"Found {len(results)} relevant chunks:\n")
    print("-" * 50)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"> Source: {result['metadata']['source']}")
        print(f"> Score: {result['score']:.3f}")
        print(f'> Text preview:\n\n{result["text"]}\n')
        print("-" * 50)

    print(f"Index Statistics: {rag.get_stats()}")
