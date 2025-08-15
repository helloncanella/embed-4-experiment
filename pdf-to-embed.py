#!/usr/bin/env python3
"""
pdf-to-embed.py

Script to convert a PDF to per-page multimodal inputs and generate/store embeddings
using Cohere Embed v4 and ChromaDB, adapted from the Cohere docs:
https://docs.cohere.com/v2/docs/semantic-search-embed#multimodal-pdf-search

Usage:
  export COHERE_API_KEY="..."
  python pdf-to-embed.py --pdf /path/to/file.pdf

"""
from pdf2image import convert_from_path
from io import BytesIO
import base64
import chromadb
import cohere
import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict


def pdf_to_image_entries(pdf_path: str, dpi: int = 200) -> List[dict]:
    """Convert a PDF into a list of input entries suitable for Cohere embed API.

    Each page becomes an entry with a small text field and a base64-encoded PNG image URL.
    """
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path="/opt/homebrew/bin")
    input_array = []
    for page in pages:
        buffer = BytesIO()
        page.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_image = f"data:image/png;base64,{base64_str}"
        page_entry = {
            "content": [
                {"type": "text", "text": f"{os.path.basename(pdf_path)}"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ]
        }
        input_array.append(page_entry)
    return input_array


def embed_pages_and_store(
    co_client: "cohere.ClientV2",
    input_array: List[dict],
    model: str,
    collection_name: str = "pdf_pages",
    persist_dir: Optional[str] = None,
) -> Tuple["chromadb.api.models.Collection", List[str]]:
    """Generate embeddings for each page and store them in a Chroma collection.
    Returns the created collection and the list of ids.
    """
    embeddings = []
    for i in range(len(input_array)):
        res = co_client.embed(
            inputs=[input_array[i]],
            model=model,
            input_type="search_document",
            embedding_types=["float"],
        ).embeddings.float[0]
        embeddings.append(res)

    ids = [str(i) for i in range(len(input_array))]

    # Use PersistentClient when a persist_dir is provided so no separate server is needed
    if persist_dir:
        # PersistentClient will manage on-disk storage (duckdb+parquet) at the given path
        chroma_client = chromadb.PersistentClient(path=persist_dir)
    else:
        chroma_client = chromadb.Client()

    # get_or_create_collection avoids race/errors if already exists
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except Exception:
        # fallback to create_collection for older chromadb versions
        collection = chroma_client.create_collection(collection_name)

    collection.add(embeddings=embeddings, ids=ids)

    # PersistentClient writes to disk automatically; attempt explicit persist if available
    if persist_dir:
        try:
            chroma_client.persist()
        except Exception:
            pass

    return collection, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF to embeddings using Cohere embed-v4 and store in ChromaDB.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF->image conversion")
    parser.add_argument("--model", default="embed-v4.0", help="Cohere embed model to use")
    parser.add_argument("--collection", default="pdf_pages", help="Chroma collection name")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Directory to persist Chroma DB (uses duckdb+parquet).")
    parser.add_argument("--query", default=None, help="Optional query to run after embedding")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return for the query")
    args = parser.parse_args()

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("Please set COHERE_API_KEY in the environment.", flush=True)
        sys.exit(1)

    co_client = cohere.ClientV2(api_key=api_key)

    print(f"Converting PDF to images: {args.pdf}", flush=True)
    input_array = pdf_to_image_entries(args.pdf, dpi=args.dpi)
    print(f"Generated {len(input_array)} page entries; embedding with model {args.model}...", flush=True)

    collection, ids = embed_pages_and_store(co_client, input_array, args.model, args.collection, persist_dir=args.persist_dir)
    print(f"Stored {len(ids)} embeddings in Chroma collection '{args.collection}'.", flush=True)
    if args.persist_dir:
        print(f"Chroma DB persisted to: {args.persist_dir}", flush=True)

    # if args.query:
    #     print(f"Running query: {args.query}")
    #     results = query_collection(co_client, collection, args.query, args.model, top_k=args.top_k)
    #     print("Top result ids:", results.get("ids"))
    # else:
    #     print("No query provided. Done.")


if __name__ == "__main__":
    main()
