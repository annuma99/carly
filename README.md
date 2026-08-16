# Carly's Pond

Carly is a local question-answering assistant for the Stevens Student Government Association (SGA) governing documents. It searches the document corpus with hybrid retrieval, then uses Claude to produce a grounded answer with citations.

The web interface is an animated pond: Carly uses the supplied GIF sprites while thinking and answering, and responses appear with a typewriter effect.

## How it works

1. PDFs are extracted into page-aware text chunks.
2. Voyage AI creates embeddings for those chunks.
3. A question is retrieved using both semantic search and BM25 keyword search, combined with reciprocal-rank fusion.
4. Claude answers from the retrieved excerpts only and saves recent conversation context for the current browser session.

## Requirements

- Python 3.10 or later
- A Voyage AI API key
- An Anthropic API key

Install the Python dependencies from the project root:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
VOYAGE_API_KEY=your_voyage_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Do not commit `.env` or API keys to source control.

## Run the web app

From `scripts/`, start the API server:

```bash
python3 api_server.py
```

Then open `scripts/index.html` in a browser. The page sends requests to `http://localhost:5001/ask`.

The animated assets expected by the interface are:

```text
assets/
├── idle_duck.gif
├── thinking_duck.gif
└── pond_background.gif
```

`pond_background.gif` is the full pond backdrop. `thinking_duck.gif` is displayed while Carly searches and generates a response; `idle_duck.gif` is used when she arrives to present it.

## Add or update PDFs

Place searchable PDFs in `data/raw/inbox/` and run:

```bash
cd scripts
python3 ingest_pdfs.py
```

Useful options:

```bash
python3 ingest_pdfs.py --dry-run  # preview imports without API calls or writes
python3 ingest_pdfs.py --force    # reprocess every inbox PDF
```

The importer is incremental. It records a SHA-256 hash per file in `data/processed/ingestion_manifest.json`, skips unchanged files, and replaces chunks only for PDFs that changed. PDFs must contain selectable text; OCR scanned PDFs before importing them.

## Project structure

```text
assets/                 Animated pond and Carly GIFs
data/raw/inbox/         PDFs waiting to be imported
data/processed/         Chunks, embeddings, and ingestion manifest
scripts/
  api_server.py         Flask endpoint for the web UI
  index.html            Animated Carly's Pond interface
  ingest_pdfs.py        Incremental PDF importer and embedder
  hybrid_search.py      BM25 + Voyage semantic retrieval
  generate_answer.py    Grounded Claude answer generation
  sessions.py           In-memory browser-session context
requirements.txt        Python dependencies
```

## Command-line use

You can also ask questions directly from the terminal:

```bash
cd scripts
python3 generate_answer.py
```

For retrieval-only inspection:

```bash
python3 hybrid_search.py
```

## Notes

- The API server and session history are designed for local development. Session history is stored only in memory and resets when the server restarts.
- The answer generator is instructed to answer only from retrieved SGA document excerpts. When the corpus does not contain enough information, it should say so rather than guess.
- Each query makes calls to Voyage AI and Anthropic, so their account limits and billing rules apply.
