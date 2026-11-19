"""Build the BioDex reaction-term Chroma index with nomic embeddings on Morfeu.

Why: every existing .chroma-biodex (tarball, /mnt/scratch/global, repo root) stores
1536-d text-embedding-3-small vectors, but biodex-demo.py embeds queries with
nomic-embed-text-v1 (768-d) through the local vLLM embedding server. Queries and
documents must use the same model, so we re-embed the 24,313 terms with the SAME
server that answers queries (port $EMBEDDING_PORT).

Usage (from abacus-research/, embedder running):
    source morfeu/env_morfeu.sh && python morfeu/build_biodex_index.py
"""
import os
import shutil
import sys

import chromadb
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from openai import OpenAI

def _find_source_index():
    for cand in [
        os.environ.get("BIODEX_SOURCE_INDEX", ""),
        "/scratch/hpc4ai/palimpzest/otimizacao-abacus/palimpzest/abacus-research/.chroma-biodex",
        "/scratch/hpc4ai/palimpzest/otimizacao-abacus/palimpzest/abacus-research/.chroma-biodex-openai",
        "/mnt/scratch/global/palimpzest/abacus-research/.chroma-biodex",
        "/scratch/global/palimpzest/abacus-research/.chroma-biodex",
        "/app/abacus-research/.chroma-biodex",
    ]:
        if cand and os.path.exists(cand):
            return cand
    return "/mnt/scratch/global/palimpzest/abacus-research/.chroma-biodex"


SOURCE_INDEX = _find_source_index()  # only used for the term list
TARGET = os.environ.get("BIODEX_CHROMA_DIR", ".chroma-biodex")
COLLECTION = "biodex-reaction-terms"
MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
PORT = os.environ.get("EMBEDDING_PORT", "8010")
BATCH = 512


def main():
    src = chromadb.PersistentClient(SOURCE_INDEX).get_collection(COLLECTION)
    terms = src.get(include=["documents"])
    ids, docs = terms["ids"], terms["documents"]
    print(f"[index] {len(docs)} reaction terms read from {SOURCE_INDEX}")

    if os.path.exists(TARGET):
        backup = TARGET.rstrip("/") + ".bak"
        shutil.rmtree(backup, ignore_errors=True)
        shutil.move(TARGET, backup)
        print(f"[index] existing {TARGET} moved to {backup}")

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{PORT}/v1")
    ef = OpenAIEmbeddingFunction(api_key="EMPTY", model_name=MODEL)  # same EF the demo uses
    col = chromadb.PersistentClient(TARGET).create_collection(
        COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    for i in range(0, len(docs), BATCH):
        chunk = docs[i : i + BATCH]
        emb = [d.embedding for d in client.embeddings.create(input=chunk, model=MODEL).data]
        col.add(ids=ids[i : i + BATCH], documents=chunk, embeddings=emb)
        print(f"\r[index] {min(i + BATCH, len(docs))}/{len(docs)}", end="", flush=True)
    print()

    # sanity check: a query should retrieve itself
    q = client.embeddings.create(input=["Drug ineffective"], model=MODEL).data[0].embedding
    hit = col.query(query_embeddings=[q], n_results=3)["documents"][0]
    print(f"[index] dim={len(q)} count={col.count()} query 'Drug ineffective' -> {hit}")
    if "Drug ineffective" not in hit:
        sys.exit("[index] sanity check FAILED")
    print(f"[index] done -> {TARGET}")


if __name__ == "__main__":
    main()
