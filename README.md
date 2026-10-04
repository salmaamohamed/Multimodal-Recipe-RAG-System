# Recipe-Only Multimodal RAG System

A Streamlit assistant that answers **only** recipe questions, grounded in the recipe PDFs
stored in `data/pdfs/`. A question may be accompanied by an image (a dish or ingredients).

```
User (question + image?) -> Recipe Query Validator -> NOT_RECIPE -> reject
                                   |
                                 RECIPE
                                   v
        BLIP caption (if image) -> embeddings -> hybrid retrieval (text + image captions)
                                   v
              recipe-level ranking -> knowledge-base check -> context -> LLM -> answer
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
copy .env.example .env            # then put your key in .env
```

The LLM is any OpenAI-compatible endpoint (`OPENAI_API_KEY` or `LLM_API_KEY`, `LLM_MODEL`,
optional `LLM_BASE_URL`). `LLM_API_KEY` wins over `OPENAI_API_KEY`, which is useful when a
different `OPENAI_API_KEY` is already set system-wide. Keep real keys in `.env` (git-ignored),
never in `.env.example`. Without a key the app still works: the validator falls back to a keyword
heuristic and answers quote the stored text of the best-matching recipe.

OCR is only a fallback for scanned pages and needs the Tesseract binary on `PATH`.

## Usage

```bash
python scripts/ingest.py     # build the knowledge base (run once, and after PDFs change)
streamlit run app.py         # the app only queries the persistent vector store
```

`scripts/ingest.py` discovers every `*.pdf` in `data/pdfs/` (no file names are hard-coded),
never modifies the PDFs, and prints a per-PDF report including anything it could not process.
It writes:

| Output | Content |
|---|---|
| `data/images/` | extracted recipe photos (`<recipe_id>_page_<n>_image_<k>.jpg`) and the BLIP caption cache |
| `data/vectorstore/` | persistent ChromaDB collection (text chunks + image-caption records) and `catalog.json` |

From Python:

```python
from src.rag.pipeline import process_query
result = process_query("How do I make Aloo Gobi?")            # text only
result = process_query("Which recipe looks like this?", image) # PIL image or path
result = process_query(None, image)                            # image only
print(result.status, result.answer, result.sources, result.images)
```

`result.status` is one of `RECIPE_FOUND`, `RECIPE_NOT_FOUND`, `NOT_RECIPE`.

## How it works

**Ingestion** (`src/ingestion/`)

- `pdf_extractor.py` — PyMuPDF text extraction with layout handling: a recursive XY-cut puts
  multi-column pages in reading order and merges ingredient/amount tables row by row;
  letter-spaced fonts (`M U S H R O O M`) are rebuilt from glyph gaps; repeated
  headers/footers/page numbers are removed; pages with no text layer fall back to Tesseract.
- `recipe_parser.py` — recipe-aware segmentation. A recipe starts at a title (text clearly
  larger than the body); title-less pages continue the previous recipe; title-only pages are
  category dividers. Lines are assigned to `description`, `ingredients`, `instructions`,
  `cooking_info`, `nutrition`, `notes`, and prep/cook/total time, servings and difficulty
  are parsed into metadata.
- `image_extractor.py` — saves embedded photos, skipping logos, repeated template artwork and
  page backgrounds, and links each to the recipe that owns its page.
- `captioner.py` — BLIP captions. They are a retrieval handle, not ingredient detection.
- `chunker.py` — one chunk per recipe section (long sections split on line boundaries), each
  prefixed with the recipe name. IDs are deterministic (`<recipe_id>::text::<section>::<i>`).

**Query time**

- `src/validation/query_validator.py` — `RECIPE` / `NOT_RECIPE`, decided before any vector
  search (LLM classifier, or keyword heuristic when no LLM is configured).
- `src/retrieval/hybrid_retriever.py` — embeds the question and/or the BLIP caption of the
  uploaded image, searches text records and image-caption records separately, de-duplicates,
  groups by `recipe_id` and scores each recipe:
  `score = text_score * TEXT_WEIGHT + image_score * IMAGE_WEIGHT` (weights renormalised when a
  recipe has hits in only one modality; a bonus when the recipe name appears in the question).
- `src/rag/kb_detector.py` — `RECIPE_FOUND` / `RECIPE_NOT_FOUND`. A recipe only counts as found
  when the dish/ingredient words of the question occur in the retrieved recipes or the match
  is strong; the LLM gives the final verdict from the context.
- `src/rag/context_builder.py` — one source-attributed block per ranked recipe (complete
  sections once, no repeated chunks), plus an ingredient index listing every recipe whose
  ingredient list mentions a word of the question, so "which recipes contain potatoes?" is
  answered across the whole knowledge base and not only the top hits.
- `src/rag/generator.py` — grounded answer with `Source: <PDF> — Page <n>` citations; never
  invents a missing recipe.

Everything tunable lives in `src/config.py` and can be overridden through environment
variables (see `.env.example`): models, paths, `TEXT_WEIGHT` / `IMAGE_WEIGHT`, top-k,
context size and the knowledge-base thresholds.

## Known limitations

- BLIP captions are generic ("a bowl of curry with rice"), so image search finds visually
  similar dishes rather than identifying a specific recipe.
- Images on pages that belong to no recipe (covers, category dividers) are not indexed.
- Cuisine is not stored as metadata because the PDFs do not state it per recipe; the
  cookbook's chapter name is kept as `category`.
- Each question is answered on its own; follow-ups such as "and how long does it take?" need
  the recipe name repeated.
