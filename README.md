# Offline RAG Pipeline for Code Documentation

A lightweight, easy to use, and completely offline Retrieval-Augmented Generation (RAG) pipeline. This tool helps you build a searchable knowledge base from your code base or documentation files without requiring internet access or external APIs.

## 🌟 Features

- **100% Offline**: Works entirely on your local machine with no internet required
- **Persistent Storage**: Saves and loads indexes from disk
- **Smart Chunking**: Intelligent structure-aware document chunking with configurable overlap
- **Hybrid Search**: Combines semantic (FAISS) and frequency-based (BM25) search for better results
- **GPU Acceleration**: Optional CUDA support for faster embeddings

## 📂 Project Structure

```
.
├── main.py                # Example usage
├── src/
│   ├── offline_rag.py     # Main pipeline code
│   ├── splitting.py       # Structure aware splitters
│   └── utils.py           # Logging and OS operations
├── tests/                 # Unit and integration tests suite
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

#### <u>With pip :</u>

```bash
pip install torch transformers numpy faiss-cpu langchain-text-splitters rank-bm25
```

#### <u>With uv :</u>

```bash
uv sync  # And you are good to go
```

### 2. Download a Model (One-Time Setup)

If you have internet access, download a model once:

```python
from transformers import AutoTokenizer, AutoModel

# Choose a model (recommended for code docs)
model_name = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
save_path = "./models/my-model"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```

Alternatively, copy pre-downloaded model files from another machine.

### 3. Basic Usage

```python
from offline_rag import OfflineCodeDocRAG

# Initialize the pipeline
rag = OfflineCodeDocRAG(
    model_path="./models/my-model",
    index_path="./my_index",
    chunk_size=2000,
    chunk_overlap=150,
    batch_size=32,
)

# Add documentation files
rag.add_files(['./docs', './README.md'])

# Search your documentation
results = rag.retrieve("How do I implement authentication?", top_k=5)

for result in results:
    print(f"Source: {result['metadata']['source']} (score: {result['score']:.2f})")
    print(f"Content: {result['text']}...")
```

## ⚠️ Important Notes

To adapt this for your specific use case, you'll certainly need to try different hyperparameter values.

<u>Parameters for RAG quality:</u>

- `chunk_size`: Higher values capture more context but may reduce precision and include irrelevant information.
- `chunk_overlap`: Controls how much context is shared between chunks. Too much overlap may cause redundancy, while too little may break up logical units.
- `semantic_weight` and `bm25_weight`: For retrieval, adjust these to balance between semantic similarity and words-frequency matching.

<u>Parameters for performance:</u>

- `batch_size`: Larger batches process faster but require more memory. Adjust based on your GPU/CPU capabilities.
- `index_type`: You can choose between "flat" (default, exact search), or more scalable options like "ivf" (cluster-based approximation) or "hnsw" (graph-based approximation).

## 📚 Detailed Usage

### Adding Documents

#### From Files and Directories
```python
# Add specific files
rag.add_files(['./api_docs.md', './guide.txt'])

# Add entire directories (recursive)
rag.add_files(['./documentation', './examples'])

# Specify file extensions
rag.add_files(['./src'], extensions=['.py', '.md', '.rst'])
```

#### Programmatically
```python
documents = [
    {
        'content': 'Your documentation content here...',
        'source': 'manual_entry.md'
    },
    {
        'content': 'API reference documentation...',
        'source': 'api_ref.md'
    }
]
rag.add_documents(documents)
```

### Retrieving Information

We use a hybrid search approach combining semantic and frequency-based search for optimal results. The FAISS (semantic) and BM25 (frequency) rankings are combined using reciprocal rank fusion.

```python
# Basic retrieval
results = rag.retrieve("query string", top_k=5)

# Different weight for semantic and bm25 search
results = rag.retrieve(
    "async functions", 
    top_k=3,
    semantic_weight=0.8,
    bm25_weight=0.2,
)

# Process results
for result in results:
    print(f"Relevance Score: {result['score']}")
    print(f"> From semantic: {result['from_semantic']}")
    print(f"> From bm25: {result['from_bm25']}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Start char: {result['metadata']['start_index']}")
    print(f"Document ID: {result['metadata']['id']}")
    print(f"Content: {result['text']}")
```

### Managing the Index

```python
# Get index statistics
stats = rag.get_stats()
print(f"Total chunks: {stats['total_documents']}")
print(f"Unique sources: {stats['unique_sources']}")
print(f"Index size: {stats['index_size_bytes']} bytes")

# Clear and rebuild index
rag.clear_index()

# Load existing index (happens automatically)
rag = OfflineCodeDocRAG(
    model_path="./models/my-model",
    index_path="./existing_index"  # Will load if exists
)
```

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model implementations
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [LangChain](https://github.com/langchain-ai/langchain) for document processing utilities
- [BM25](https://github.com/dorianbrown/rank_bm25) for frequency-based ranking

