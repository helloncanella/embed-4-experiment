# import chromadb
# import cohere


# from typing import Dict, List


# def query_collection(co_client: cohere.ClientV2, collection: chromadb.api.models.Collection, queries: str, model: str, top_k: int = 5) -> Dict[str, List]:
#     query_input = [{"content": [{"type": "text", "text": query}]} for query in queries]

#     query_emb = co_client.embed(
#         inputs=query_input,
#         model=model,
#         input_type="search_query",
#         embedding_types=["float"],
#     ).embeddings

#     query_emb = [emb.float[0] for emb in query_emb]

#     results = collection.query(query_embeddings=query_emb, n_results=top_k)

#     return results


# import asyncio

# if __name__ == "__main__":
#     import os

#     api_key = os.environ.get("COHERE_API_KEY")
#     co_client = cohere.ClientV2(api_key=api_key)
#     collection = chromadb.PersistentClient(path="./chroma_db").get_collection("pdf_pages")

#     j = query_collection(co_client, collection, ["montanha", "impressum"], "embed-v4.0", top_k=5)

#     print(j)
#     # async def main():
#     # api_key = os.environ.get("COHERE_API_KEY")

#     # asyncio.run(main())

import chromadb
import cohere
from typing import Dict, List, Any


def query_collection(
    co_client: cohere.ClientV2, collection: chromadb.api.models.Collection, queries: List[str], model: str, top_k: int = 5  # <- era str; agora é List[str]
) -> Dict[str, List[Any]]:
    # Gere embeddings de texto (query) em float
    res = co_client.embed(
        texts=queries,  # <- use texts para texto puro
        model="embed-v4.0",
        input_type="search_query",  # tipo otimizado para consultas
        embedding_types=["float"],  # pedimos floats
    )

    # Vetores já vêm como List[List[float]]
    query_embs: List[List[float]] = res.embeddings.float  # <- acesso correto

    # Busque no Chroma com várias queries de uma vez
    results = collection.query(query_embeddings=query_embs, n_results=top_k)
    return results.get("ids")


# if __name__ == "__main__":
#     import os

#     co_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
#     collection = chromadb.PersistentClient(path="./chroma_db").get_collection("pdf_pages")

#     out = query_collection(
#         co_client,
#         collection,
#         ["fluss", "impressum"],
#         model="embed-v4.0",
#         top_k=5,
#     )

#     print(out)

from typing import Iterable, List, Tuple


def expand_pages(pages: Iterable[str | int], min_page: int = 0, max_page: int = 167) -> List[Tuple[int, ...]]:
    """Expande cada página para (n-1, n, n+1), cortando no limite [min_page, max_page]."""
    out: List[Tuple[int, ...]] = []
    for p in pages:
        n = int(p)
        start = max(n - 1, min_page)
        end = min(n + 1, max_page)
        out.append(tuple(range(start, end + 1)))
    return out


def all_pages(pages: List[Iterable[str | int]], min_page: int = 0, max_page: int = 167) -> List[Tuple[int, ...]]:
    return [expand_pages(page, min_page, max_page) for page in pages]


def pair_search(co_client: cohere.ClientV2, collection: chromadb.api.models.Collection, queries: List[str], model: str, top_k: int = 5) -> Dict[str, List[Any]]:
    # zipped_pages = all_pages(queries)

    result = query_collection(co_client, collection, queries, model, top_k)

    zipped_pages = zip(queries, all_pages(result))

    print(list(zipped_pages))

    # pass


# # exemplos:
# first = ["2", "0", "49", "24", "126"]
# second = ["166", "127", "22", "3", "141"]

# print(all_pages([first, second]))

if __name__ == "__main__":
    import os

    api_key = os.environ.get("COHERE_API_KEY")
    co_client = cohere.ClientV2(api_key=api_key)
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("pdf_pages")

    pair_search(co_client, collection, ["fluss", "impressum"], "embed-v4.0", top_k=5)
