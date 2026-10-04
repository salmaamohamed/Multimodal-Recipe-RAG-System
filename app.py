"""Streamlit UI for the recipe-only multimodal RAG assistant.

    streamlit run app.py

The app only queries the persistent vector store; build it first with
`python scripts/ingest.py`.
"""
from __future__ import annotations

import streamlit as st
from PIL import Image

from src.config import ROOT, settings
from src.rag.pipeline import KnowledgeBaseMissing, QueryResult, get_retriever, process_query
from src.rag.llm import get_llm
from src.retrieval.catalog import get_catalog

st.set_page_config(page_title="Recipe Assistant", page_icon="🍲", layout="wide")

STATUS_BADGE = {
    "RECIPE_FOUND": ("Recipe found", "green"),
    "RECIPE_NOT_FOUND": ("Not in knowledge base", "orange"),
    "NOT_RECIPE": ("Not a recipe question", "red"),
}


@st.cache_resource(show_spinner="Loading the recipe knowledge base ...")
def load_retriever():
    return get_retriever()


def render_result(result: QueryResult) -> None:
    label, color = STATUS_BADGE[result.status]
    st.markdown(f":{color}-badge[{label}]")
    st.markdown(result.answer)

    shown = [r for r in result.used if r.images]
    if shown:
        cols = st.columns(min(len(shown), 3))
        for col, recipe in zip(cols, shown[:3]):
            path = ROOT / recipe.images[0]
            if path.exists():
                col.image(str(path), caption=f"{recipe.recipe_name} — {recipe.source_pdf}",
                          width="stretch")

    if result.sources:
        with st.expander("Sources"):
            for src in result.sources:
                pages = ", ".join(map(str, src["pages"]))
                st.markdown(f"- **{src['recipe_name']}** — {src['source_pdf']}, page {pages}")

    with st.expander("Retrieval details"):
        if result.validation:
            st.write(f"Validator: **{result.validation.label}** "
                     f"({result.validation.method}) {result.validation.reason}")
        if result.image_caption:
            st.write(f"Image caption (BLIP): _{result.image_caption}_")
        if result.evidence:
            st.write(f"Knowledge base check: **{result.evidence.status}** — {result.evidence.reason}")
        if result.recipes:
            st.dataframe(
                [{"recipe": r.recipe_name, "score": r.score, "text": round(r.text_score, 3),
                  "image": round(r.image_score, 3), "source": r.source_pdf,
                  "pages": ", ".join(map(str, r.pages))} for r in result.recipes[:8]],
                hide_index=True, width="stretch")
        if result.context:
            st.text_area("Context sent to the LLM", result.context, height=240)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Knowledge base")
    try:
        load_retriever()
        catalog = get_catalog()
        sources = sorted({r["source_pdf"] for r in catalog.recipes})
        st.metric("Recipes", len(catalog))
        st.caption(f"from {len(sources)} PDF documents")
        with st.expander("Browse recipes"):
            for pdf in sources:
                names = [r["recipe_name"] for r in catalog.recipes if r["source_pdf"] == pdf]
                st.markdown(f"**{pdf}** ({len(names)})")
                st.caption(" · ".join(names))
        kb_ready = True
    except KnowledgeBaseMissing as exc:
        st.error(str(exc))
        kb_ready = False

    st.divider()
    st.header("Image (optional)")
    upload = st.file_uploader("Dish or ingredients photo", type=["jpg", "jpeg", "png", "webp"])
    image = Image.open(upload) if upload else None
    if image:
        st.image(image, width="stretch")
        image_only = st.button("Find recipes for this image", width="stretch")
    else:
        image_only = False

    st.divider()
    llm = get_llm()
    st.caption(f"LLM: {llm.model}" if llm.available
               else "No LLM key set — answers quote the stored recipe text.")
    st.caption(f"Embeddings: {settings.embedding_model}")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.pop("messages", None)
        st.rerun()

# --------------------------------------------------------------------- chat
st.title("🍲 Recipe Assistant")
st.caption("Ask about the recipes in the knowledge base — ingredients, steps, cooking times, "
           "comparisons, or what you can cook. Attach a photo in the sidebar to search by image.")

messages = st.session_state.setdefault("messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["text"])
            if message.get("image") is not None:
                st.image(message["image"], width=220)
        else:
            render_result(message["result"])

question = st.chat_input("Ask a recipe question ...", disabled=not kb_ready)
if kb_ready and (question or image_only):
    messages.append({"role": "user", "text": question or "_(image only)_", "image": image})
    with st.chat_message("user"):
        st.markdown(question or "_(image only)_")
        if image is not None:
            st.image(image, width=220)
    with st.chat_message("assistant"):
        with st.spinner("Searching the recipes ..."):
            result = process_query(question, image)
        render_result(result)
    messages.append({"role": "assistant", "result": result})
