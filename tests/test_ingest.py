"""Unit tests for ingest.py.

Pure functions (load_file, discover) are tested directly. main() is tested with
the embedding model and Milvus connection mocked, so no GPU/network/Milvus is
needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ingest


# --------------------------------------------------------------------------- #
# load_file                                                                    #
# --------------------------------------------------------------------------- #
def test_load_text_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\nsome body", encoding="utf-8")

    docs = ingest.load_file(f, tmp_path)

    assert len(docs) == 1
    assert docs[0].page_content == "# Title\nsome body"
    assert docs[0].metadata == {"source": "note.md"}


def test_load_text_file_source_is_relative(tmp_path):
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    f = sub / "deep.txt"
    f.write_text("hi", encoding="utf-8")

    docs = ingest.load_file(f, tmp_path)

    assert docs[0].metadata["source"] == str(Path("a/b/deep.txt"))


def test_load_pdf_file_skips_empty_pages(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("page one"), FakePage("  "), FakePage("page three")]

    monkeypatch.setattr(ingest, "PdfReader", FakeReader)

    docs = ingest.load_file(tmp_path / "doc.pdf", tmp_path)

    # Empty/whitespace page is dropped; remaining keep their original page numbers.
    assert [d.page_content for d in docs] == ["page one", "page three"]
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert all(d.metadata["source"] == "doc.pdf" for d in docs)


def test_load_unsupported_file_returns_empty(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b", encoding="utf-8")

    assert ingest.load_file(f, tmp_path) == []


# --------------------------------------------------------------------------- #
# discover                                                                     #
# --------------------------------------------------------------------------- #
def test_discover_finds_supported_sorted_and_ignores_others(tmp_path):
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.log").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.pdf").write_text("x", encoding="utf-8")

    found = [p.relative_to(tmp_path).as_posix() for p in ingest.discover(tmp_path)]

    assert found == ["a.txt", "b.md", "sub/c.pdf"]


def test_discover_empty_dir(tmp_path):
    assert ingest.discover(tmp_path) == []


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
class FakeStore:
    def __init__(self):
        self.added = None

    def add_documents(self, docs):
        self.added = docs


def _patch_backend(monkeypatch):
    """Mock the embedding + vector store layer; return the captured state."""
    state = {"store": FakeStore(), "vs_kwargs": None}
    monkeypatch.setattr(ingest, "get_embeddings", lambda: "FAKE_EMB")

    def fake_get_vectorstore(**kwargs):
        state["vs_kwargs"] = kwargs
        return state["store"]

    monkeypatch.setattr(ingest, "get_vectorstore", fake_get_vectorstore)
    return state


def test_main_missing_folder_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.sys, "argv", ["ingest.py", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        ingest.main()


def test_main_empty_folder_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.sys, "argv", ["ingest.py", str(tmp_path)])
    with pytest.raises(SystemExit):
        ingest.main()


def test_main_happy_path_rebuilds(tmp_path, monkeypatch):
    (tmp_path / "doc.md").write_text("ACME deploys on Tuesdays.", encoding="utf-8")
    state = _patch_backend(monkeypatch)
    monkeypatch.setattr(ingest.sys, "argv", ["ingest.py", str(tmp_path)])

    ingest.main()

    # Default run rebuilds (drop_old=True) and writes chunks to the store.
    assert state["vs_kwargs"]["drop_old"] is True
    assert state["vs_kwargs"]["embeddings"] == "FAKE_EMB"
    assert state["store"].added is not None and len(state["store"].added) >= 1
    assert "ACME" in state["store"].added[0].page_content


def test_main_append_does_not_drop(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("hello world", encoding="utf-8")
    state = _patch_backend(monkeypatch)
    monkeypatch.setattr(ingest.sys, "argv", ["ingest.py", str(tmp_path), "--append"])

    ingest.main()

    assert state["vs_kwargs"]["drop_old"] is False
    assert state["store"].added is not None
