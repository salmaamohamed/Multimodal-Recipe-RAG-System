"""Build the recipe knowledge base from the PDFs in data/pdfs.

    python scripts/ingest.py              # full rebuild
    python scripts/ingest.py --no-images  # text only (skips BLIP)

PDFs are only read, never modified. The Streamlit app queries the result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.ingestion.chunker import image_records, text_records  # noqa: E402
from src.ingestion.image_extractor import extract_images  # noqa: E402
from src.ingestion.pdf_extractor import extract_document  # noqa: E402
from src.ingestion.recipe_parser import parse_recipes  # noqa: E402


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf-dir", type=Path, default=settings.pdf_dir)
    parser.add_argument("--no-images", action="store_true", help="skip image extraction and BLIP")
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if not pdfs:
        print(f"No PDF files found in {args.pdf_dir}")
        return 1
    print(f"Found {len(pdfs)} PDF(s) in {args.pdf_dir}\n")
    started = time.time()

    # generated images are rebuilt from scratch; the caption cache is kept
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    for old in settings.image_dir.glob("*.jpg"):
        old.unlink()

    records: list[dict] = []
    catalog: list[dict] = []
    all_images = []
    failed: list[str] = []

    for pdf in pdfs:
        try:
            doc = extract_document(pdf)
            recipes, owner, notes = parse_recipes(doc)
        except Exception as exc:
            failed.append(f"{pdf.name}: {exc}")
            print(f"[FAILED] {pdf.name}: {exc}")
            continue
        images, image_notes = ([], []) if args.no_images else extract_images(
            pdf, owner, recipes, settings.image_dir)
        all_images.extend(images)

        ocr_pages = sum(p.ocr_used for p in doc.pages)
        print(f"{pdf.name}\n  pages: {len(doc.pages)}  recipes: {len(recipes)}  "
              f"images: {len(images)}  OCR pages: {ocr_pages}")
        for note in doc.warnings + notes + image_notes:
            print(f"  - {note}")
        if not recipes:
            failed.append(f"{pdf.name}: no recipe could be identified")

        for recipe in recipes:
            chunks = text_records(recipe)
            records.extend(chunks)
            catalog.append({
                "recipe_id": recipe.recipe_id, "recipe_name": recipe.name,
                "source_pdf": recipe.source_pdf, "category": recipe.category,
                "pages": recipe.pages, "info": recipe.info,
                "sections": [s for s in recipe.sections if recipe.sections[s]],
                "images": [i.rel_path for i in images if i.recipe is recipe],
                "tokens": sorted(tokens(recipe.name) | set().union(
                    *(tokens(c["content"]) for c in chunks))),
                "ingredient_tokens": sorted(tokens(recipe.text("ingredients"))),
            })

    if not records:
        print("Nothing to index.")
        return 1

    if all_images:
        from src.ingestion.captioner import CaptionCache
        print(f"\nCaptioning {len(all_images)} image(s) with {settings.caption_model} ...")
        cache = CaptionCache(settings.image_dir / "captions.json")
        for n, img in enumerate(all_images, 1):
            img.caption = cache.caption(img.path)
            if n % 10 == 0 or n == len(all_images):
                print(f"  {n}/{len(all_images)}")
                cache.save()
        records.extend(image_records(all_images))

    from src.embeddings.embedder import get_embedder
    from src.vectorstore.chroma_store import RecipeStore
    print(f"\nEmbedding {len(records)} record(s) with {settings.embedding_model} ...")
    embeddings = get_embedder().embed_documents([r["content"] for r in records])
    store = RecipeStore()
    store.reset(settings.embedding_model)
    store.upsert(records, embeddings)
    settings.catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

    n_images = sum(r["metadata"]["modality"] == "image" for r in records)
    print(f"\nDone in {time.time() - started:.0f}s")
    print(f"  PDFs processed : {len(pdfs) - len([f for f in failed if 'no recipe' not in f])}/{len(pdfs)}")
    print(f"  recipes        : {len(catalog)}")
    print(f"  text chunks    : {len(records) - n_images}")
    print(f"  image captions : {n_images}")
    print(f"  vector store   : {settings.vectorstore_dir} ({store.count()} records)")
    if failed:
        print("\nPDFs that could not be processed normally:")
        for item in failed:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
