"""
Ingestion pipeline — load private docs into the Milvus knowledge base.

Discovers files under ./data (recursively), splits them into chunks, embeds them
with the local model, and upserts them into the Milvus collection. Run this once
(or whenever the docs change) before querying the agent.

Usage
-----
    uv run ingest.py                 # ingest ./data, rebuild the collection
    uv run ingest.py path/to/docs    # ingest a different folder
    uv run ingest.py --append        # add to the existing collection (no drop)

Supported file types: .txt, .md, .markdown, .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from kb import COLLECTION, MILVUS_URI, get_embeddings, get_vectorstore

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES


def load_file(path: Path, root: Path) -> list[Document]:
    """Load one file into Document(s), tagged with its relative source path."""
    source = str(path.relative_to(root))
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": source, "page": 1})]

    if suffix in PDF_SUFFIXES:
        docs: list[Document] = []
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": page_num},
                    )
                )
        return docs

    return []


def discover(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def main() -> None:
    args = [a for a in sys.argv[1:]]
    append = "--append" in args
    args = [a for a in args if a != "--append"]
    root = Path(args[0]) if args else Path("data")

    if not root.exists():
        sys.exit(
            f"Source folder '{root}' does not exist. Create it and add documents "
            f"({', '.join(sorted(SUPPORTED))})."
        )

    files = discover(root)
    if not files:
        sys.exit(f"No supported documents found under '{root}'.")

    print(f"📂 Found {len(files)} file(s) under '{root}':")
    for f in files:
        print(f"   - {f.relative_to(root)}")

    docs: list[Document] = []
    for f in files:
        docs.extend(load_file(f, root))
    print(f"📄 Loaded {len(docs)} document section(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunk(s).")
    if not chunks:
        sys.exit("Nothing to ingest (documents produced no text).")

    print(f"🧠 Loading embedding model and connecting to Milvus at {MILVUS_URI} ...")
    embeddings = get_embeddings()
    # drop_old rebuilds from scratch unless --append was passed.
    vectorstore = get_vectorstore(embeddings=embeddings, drop_old=not append)
    vectorstore.add_documents(chunks)

    mode = "appended to" if append else "rebuilt"
    print(
        f"✅ Done. {mode} collection '{COLLECTION}' with {len(chunks)} vectors."
    )


if __name__ == "__main__":
    main()
