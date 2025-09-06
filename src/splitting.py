"""Module for creating splitters based on file type."""

from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

EXTENSION_TO_LANG = {
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".py": Language.PYTHON,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".html": Language.HTML,
}

DEFAULT_EXTENSIONS = list(EXTENSION_TO_LANG.keys())


def make_splitter_for_file(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter for a given file based on its extension.

    Args:
        file_path (Path): Path to the file to chunk.
        chunk_size (int): Size of text chunks in characters.
        chunk_overlap (int): Overlap between chunks in characters.

    Returns:
        A RecursiveCharacterTextSplitter for the given file type.
    """
    extension = file_path.suffix.lower()
    language = EXTENSION_TO_LANG.get(extension)

    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            keep_separator=True,
        )

    # Fallback: just generic recursive text splitting
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        keep_separator=True,
    )
