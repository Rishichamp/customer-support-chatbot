"""
Document ingestion and RAG retrieval for the Customer Support Chatbot.
Supports .md, .txt, and .pdf files from the configured DOCS_PATH.
"""
import re, json, hashlib, logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name="support_docs",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.embeddings.create(
        input=[t if t.strip() else "empty" for t in texts],
        model=settings.EMBEDDING_MODEL
    )
    return [d.embedding for d in resp.data]


def _recursive_chunk(text: str) -> list[str]:
    """Recursive character splitting with overlap."""
    cs = settings.CHUNK_SIZE * 4  # approx chars
    co = settings.CHUNK_OVERLAP * 4
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(txt, seps):
        if not seps or len(txt) <= cs:
            return [txt.strip()] if txt.strip() else []
        if seps[0] == "":
            # No natural boundary left (str.split("") raises), so cut at the size limit.
            return [txt[i:i + cs].strip() for i in range(0, len(txt), cs) if txt[i:i + cs].strip()]
        parts = txt.split(seps[0])
        chunks, cur = [], ""
        for p in parts:
            candidate = cur + (seps[0] if cur else "") + p
            if len(candidate) <= cs:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur.strip())
                chunks.extend(_split(p, seps[1:]) if len(p) > cs else [p.strip()] if p.strip() else [])
                cur = ""
        if cur:
            chunks.append(cur.strip())
        return chunks

    raw = _split(text, separators)

    # Apply overlap
    if len(raw) <= 1 or co <= 0:
        return [c for c in raw if c]
    overlapped = [raw[0]]
    for chunk in raw[1:]:
        prev_tail = overlapped[-1][-co:] if len(overlapped[-1]) > co else overlapped[-1]
        overlapped.append(prev_tail + " " + chunk)
    return [c.strip() for c in overlapped if c.strip()]


def ingest_text(text: str, doc_id: str, title: str,
                category: str = "general", tags: list = None) -> int:
    """Ingest a text document into the vector store."""
    collection = get_collection()
    chunks = _recursive_chunk(text)
    if not chunks:
        return 0
    embeddings = _get_embeddings(chunks)
    ids = [f"{doc_id}_c{i}" for i in range(len(chunks))]
    metas = [{
        "doc_id": doc_id, "title": title,
        "category": category, "chunk_idx": i,
        "tags": json.dumps(tags or []),
    } for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
    logger.info(f"Ingested '{title}': {len(chunks)} chunks")
    return len(chunks)


def ingest_directory() -> dict:
    results = {"success": [], "failed": []}
    for path in Path(settings.DOCS_PATH).rglob("*"):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            doc_id = hashlib.md5(str(path).encode()).hexdigest()[:10]
            n = ingest_text(text, doc_id, path.stem.replace("_"," ").title(), category="support")
            results["success"].append({"file": path.name, "chunks": n})
        except Exception as e:
            results["failed"].append({"file": path.name, "error": str(e)})
    return results


def retrieve(query: str, top_k: int = None,
             category_filter: Optional[str] = None) -> list[dict]:
    """Retrieve most semantically similar chunks for a query."""
    k = top_k or settings.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        return []
    q_emb = _get_embeddings([query])[0]
    kwargs = dict(
        query_embeddings=[q_emb],
        n_results=min(k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )
    if category_filter:
        kwargs["where"] = {"category": category_filter}
    results = collection.query(**kwargs)
    docs = []
    for i in range(len(results["ids"][0])):
        score = round(1.0 - results["distances"][0][i], 4)
        if score >= settings.SIMILARITY_THRESHOLD:
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
            })
    return sorted(docs, key=lambda d: d["score"], reverse=True)


def get_stats() -> dict:
    return {
        "total_chunks": get_collection().count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
    }
