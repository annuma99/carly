"""Incrementally add PDFs from ../data/raw/inbox to the Carly corpus.

Usage:
    python3 ingest_pdfs.py
    python3 ingest_pdfs.py --dry-run

The command is safe to rerun.  It records a SHA-256 hash for each imported
file, so unchanged PDFs are skipped; replacing a PDF reindexes only that PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from pypdf import PdfReader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INBOX_DIR = PROJECT_DIR / "data" / "raw" / "inbox"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "chunks_with_embeddings.json"
MANIFEST_PATH = PROCESSED_DIR / "ingestion_manifest.json"
MODEL = "voyage-3"
DEFAULT_BATCH_SIZE = 10
DEFAULT_DELAY_SECONDS = 21


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, value) -> None:
    """Write through a temporary file so an interrupted run cannot corrupt the corpus."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Normalize extraction artifacts while retaining paragraph boundaries where possible."""
    text = text.replace("\u00ad", "")  # soft hyphen inserted by some PDF generators
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)  # join line-break hyphenation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def title_for(pdf_path: Path, reader: PdfReader) -> str:
    metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    if metadata_title and metadata_title.strip():
        return clean_text(metadata_title)
    return re.sub(r"[_-]+", " ", pdf_path.stem.lstrip("#")).strip()


def split_text(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    """Make readable, bounded chunks without splitting a paragraph unless necessary."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def add(piece: str) -> None:
        nonlocal current
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_characters:
            current = candidate
            return
        if current:
            chunks.append(current)
        current = piece

    for paragraph in paragraphs:
        if len(paragraph) <= max_characters:
            add(paragraph)
            continue

        # A long paragraph is usually an extraction artifact.  Split on words
        # and retain a small tail as overlap to avoid losing context at seams.
        if current:
            chunks.append(current)
            current = ""
        words = paragraph.split()
        piece = ""
        for word in words:
            candidate = f"{piece} {word}".strip()
            if len(candidate) <= max_characters:
                piece = candidate
                continue
            chunks.append(piece)
            overlap = piece[-overlap_characters:].split(maxsplit=1)
            prefix = overlap[-1] if overlap else ""
            piece = f"{prefix} {word}".strip()
        if piece:
            current = piece

    if current:
        chunks.append(current)
    return chunks


def chunks_from_pdf(pdf_path: Path, relative_source: str, digest: str, max_characters: int) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    title = title_for(pdf_path, reader)
    extracted: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        page_chunks = split_text(text, max_characters, 200)
        for part_number, part in enumerate(page_chunks, start=1):
            citation = f"{title}, p. {page_number}"
            if len(page_chunks) > 1:
                citation += f", chunk {part_number}"
            extracted.append({
                "document": title,
                # These fields preserve compatibility with the existing corpus.
                "article": title,
                "section": f"p. {page_number}",
                "citation": citation,
                "text": part,
                "source_file": relative_source,
                "source_sha256": digest,
                "page_start": page_number,
                "page_end": page_number,
            })
    return extracted


def embed_chunks(client, chunks: list[dict], batch_size: int, delay_seconds: float) -> list[dict]:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        result = client.embed([chunk["text"] for chunk in batch], model=MODEL, input_type="document")
        for chunk, embedding in zip(batch, result.embeddings):
            chunk["embedding"] = embedding
        print(f"Embedded chunks {start + 1}-{start + len(batch)} of {len(chunks)}")
        if start + batch_size < len(chunks):
            time.sleep(delay_seconds)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Import new or changed PDFs into Carly's searchable corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without calling Voyage or writing files.")
    parser.add_argument("--force", action="store_true", help="Reprocess every PDF in the inbox, even if unchanged.")
    parser.add_argument("--chunk-size", type=int, default=1_200, help="Maximum characters per chunk (default: 1200).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embeddings per API request (default: 10).")
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="Pause between embedding batches (default: 21).")
    args = parser.parse_args()
    if args.chunk_size < 300 or args.batch_size < 1 or args.delay_seconds < 0:
        parser.error("chunk size must be at least 300; batch size and delay cannot be negative")

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(path for path in INBOX_DIR.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found. Add PDFs under {INBOX_DIR} and run this command again.")
        return

    raw_chunks = read_json(CHUNKS_PATH, [])
    embedded_chunks = read_json(EMBEDDINGS_PATH, [])
    manifest = read_json(MANIFEST_PATH, {"version": 1, "files": {}})
    manifest.setdefault("files", {})

    jobs: list[tuple[Path, str, str]] = []
    for pdf_path in pdfs:
        relative_source = pdf_path.relative_to(INBOX_DIR).as_posix()
        digest = file_hash(pdf_path)
        previous = manifest["files"].get(relative_source, {})
        if not args.force and previous.get("sha256") == digest and previous.get("status") == "complete":
            print(f"Unchanged: {relative_source}")
        else:
            jobs.append((pdf_path, relative_source, digest))

    if not jobs:
        print("Corpus is already up to date.")
        return
    print(f"{len(jobs)} PDF(s) will be imported.")
    if args.dry_run:
        for _, source, _ in jobs:
            print(f"Would import: {source}")
        return

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not found -- add it to the project .env file")
    client = voyageai.Client(api_key=api_key)

    for pdf_path, source, digest in jobs:
        new_chunks = chunks_from_pdf(pdf_path, source, digest, args.chunk_size)
        if not new_chunks:
            print(f"No selectable text found in {source}; skipped. Use OCR before importing scanned PDFs.")
            manifest["files"][source] = {"sha256": digest, "status": "no_text", "updated_at": datetime.now(timezone.utc).isoformat()}
            continue
        print(f"Extracted {len(new_chunks)} chunks from {source}")
        embedded_new_chunks = embed_chunks(client, new_chunks, args.batch_size, args.delay_seconds)

        # Replacing a file replaces its earlier imported chunks but never alters
        # the original hand-curated corpus, which has no source_file field.
        raw_chunks = [chunk for chunk in raw_chunks if chunk.get("source_file") != source]
        embedded_chunks = [chunk for chunk in embedded_chunks if chunk.get("source_file") != source]
        raw_chunks.extend([{key: value for key, value in chunk.items() if key != "embedding"} for chunk in embedded_new_chunks])
        embedded_chunks.extend(embedded_new_chunks)
        manifest["files"][source] = {
            "sha256": digest,
            "status": "complete",
            "chunk_count": len(new_chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    write_json(CHUNKS_PATH, raw_chunks)
    write_json(EMBEDDINGS_PATH, embedded_chunks)
    write_json(MANIFEST_PATH, manifest)
    print(f"Saved {len(raw_chunks)} text chunks and {len(embedded_chunks)} embedded chunks.")


if __name__ == "__main__":
    main()
