import argparse

from query_collection import query_collection


parser = argparse.ArgumentParser(description="Convert PDF to embeddings using Cohere embed-v4 and store in ChromaDB.")


from pdf2image import convert_from_path
from io import BytesIO
import base64
import chromadb
import cohere
import argparse
import os
from typing import Tuple, Optional


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert PDF to embeddings using Cohere embed-v4 and store in ChromaDB.")
    parser.add_argument("--query", required=True, help="Query to run")
    args = parser.parse_args()

    api_key = os.environ.get("COHERE_API_KEY")
    co_client = cohere.ClientV2(api_key=api_key)
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("pdf_pages")

    if args.query:

        print(f"Running query: {args.query}")
        results = query_collection(co_client, collection, args.query, "embed-v4.0", top_k=5)
        print("Top result ids:", results.get("ids"))
    else:
        print("No query provided. Done.")

    # asyncio.run(main())

    # if args.query:
    #     print(f"Running query: {args.query}")
    #     results = query_collection(co_client, collection, args.query, args.model, top_k=args.top_k)
    #     print("Top result ids:", results.get("ids"))
    # else:
    #     print("No query provided. Done.")
