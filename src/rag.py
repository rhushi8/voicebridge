import json
import math
from pathlib import Path

import requests

POLICY_DIR = Path(__file__).parent.parent / "data" / "policies"
INDEX_FILE = Path(__file__).parent.parent / "data" / "policy_index.json"

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"


def load_policy_chunks():
    chunks = []
    for md_file in sorted(POLICY_DIR.glob("*.md")):
        doc_title = md_file.stem
        heading = None
        body_lines = []
        for line in md_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("# ") and not line.startswith("## "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if heading is not None:
                    chunks.append(_chunk(doc_title, heading, body_lines))
                heading = line[3:].strip()
                body_lines = []
            else:
                body_lines.append(line)
        if heading is not None:
            chunks.append(_chunk(doc_title, heading, body_lines))
    return chunks


def _chunk(doc_title, heading, body_lines):
    return {
        "doc": doc_title,
        "heading": heading,
        "text": "\n".join(body_lines).strip(),
    }


def embed(api_key, text):
    body = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
    }
    response = requests.post(
        EMBED_URL, headers={"x-goog-api-key": api_key}, json=body, timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


def build_index(api_key):
    chunks = load_policy_chunks()
    if INDEX_FILE.exists():
        cached = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        key = lambda c: (c["doc"], c["heading"], c["text"])
        if [key(c) for c in cached] == [key(c) for c in chunks]:
            return cached

    for chunk in chunks:
        chunk["embedding"] = embed(api_key, f"{chunk['heading']}\n{chunk['text']}")
    INDEX_FILE.write_text(json.dumps(chunks), encoding="utf-8")
    return chunks


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def retrieve(api_key, index, query, top_k=2, min_score=0.6):
    query_vec = embed(api_key, query)
    scored = [(cosine(query_vec, c["embedding"]), c) for c in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for score, c in scored[:top_k] if score >= min_score]


def format_policy_context(chunks):
    if not chunks:
        return "(none retrieved for this turn)"
    parts = []
    for chunk in chunks:
        parts.append(f'From "{chunk["doc"]}", section "{chunk["heading"]}":\n{chunk["text"]}')
    return "\n\n".join(parts)
