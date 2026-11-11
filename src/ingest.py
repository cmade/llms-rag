"""Ingest Markdown docs and HTML tickets into Pinecone (v1)."""
import argparse
import hashlib
import os
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "checkpoint-rag")
NAMESPACE = "v1"
CHUNK_SIZE = 500        # tokens: max chunk size (tickets, oversized sections)
CHUNK_OVERLAP = 75      # tokens: ~15% overlap
MD_PACK_TOKENS = 250    # pack small Markdown sections together up to this size
MIN_TAIL_TOKENS = 80    # merge a small final group into the previous one
EMBED_BATCH = 100       # texts per embedding request
UPSERT_BATCH = 50       # records per Pinecone upsert (keeps requests under size limits)

enc = tiktoken.get_encoding("cl100k_base")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def norm(text: str) -> str:
    """Collapse whitespace and fix 'SSO , has' -> 'SSO, has' left behind by inline tags."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?)])", r"\1", text)


def rel_source(path: Path, corpus: Path) -> str:
    """'product-docs/admin/sso.md': matches expected_sources exactly."""
    return path.relative_to(corpus).as_posix()

FIELD_RE = re.compile(
    r"(Status|Category|Priority|Customer|Agent):\s*(.*?)"
    r"(?=\s+(?:Status|Category|Priority|Customer|Agent):|$)"
)

def split_markdown_sections(md: str):
    """Return (title, [(heading, body), ...]) split on '## ' headings."""
    title, sections = "", []
    heading, body = None, []
    for line in md.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            if heading or "".join(body).strip():
                sections.append((heading or "Overview", "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    if heading or "".join(body).strip():
        sections.append((heading or "Overview", "\n".join(body).strip()))
    return title, sections


def chunk_markdown(path: Path, corpus: Path) -> list[dict]:
    title, sections = split_markdown_sections(path.read_text(encoding="utf-8"))

    # Pack adjacent sections until adding the next would exceed MD_PACK_TOKENS
    groups, current, current_tokens = [], [], 0
    for heading, body in sections:
        t = n_tokens(body)
        if current and current_tokens + t > MD_PACK_TOKENS:
            groups.append(current)
            current, current_tokens = [], 0
        current.append((heading, body))
        current_tokens += t
    if current:
        groups.append(current)

    # NEW: fold a tiny leftover group into the previous one
        if len(groups) > 1 and sum(n_tokens(b) for _, b in groups[-1]) < MIN_TAIL_TOKENS:
            tail = groups.pop()      # remove the small last group first...
            groups[-1] += tail       # ...then add it to what is now the last group

    chunks = []
    for group in groups:
        headings = [h for h, _ in group]
        body = "\n\n".join(f"## {h}\n{b}" for h, b in group)
        crumb = f"{title} > {'; '.join(headings)}"
        # Safety net: recursively split only an oversized group
        pieces = splitter.split_text(body) if n_tokens(body) > CHUNK_SIZE else [body]
        for piece in pieces:
            chunks.append({
                "text": f"{crumb}\n\n{piece}",
                "source": rel_source(path, corpus),
                "doc_type": "product-docs",
                "title": title,
                "section": "; ".join(headings),
            })
    return chunks

def _text(el) -> str:
    return norm(el.get_text(" ", strip=True)) if el else ""


def clean_ticket(raw: str):
    """Return (ticket_id, title, fields, clean_text) from a ticket's HTML."""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "head"]):      # head holds the duplicate <title>
        tag.decompose()

    title = _text(soup.find("h1"))
    ticket_id = _text(soup.select_one(".ticket-id")).replace("TICKET", "").strip()

    # NEW: match known labels instead of splitting at the first colon
    meta_text = " ".join(_text(row) for row in soup.select(".meta-row"))
    fields = {label.lower(): value.strip() for label, value in FIELD_RE.findall(meta_text)}

    parts = [
        f"Ticket {ticket_id}: {title}",
        " | ".join(f"{k.title()}: {v}" for k, v in fields.items()),
    ]
    for msg in soup.select(".message"):
        who = (f"[{_text(msg.select_one('.role-badge'))}] "
               f"{_text(msg.select_one('.sender-name'))} "
               f"({_text(msg.select_one('.timestamp'))}):")
        body = msg.select_one(".message-body")
        # Innermost <p>/<li> only, so a <p> inside an <li> isn't counted twice
        blocks = [b for b in body.find_all(["p", "li"]) if not b.find(["p", "li"])] if body else []
        lines = [("- " if b.name == "li" else "") + _text(b) for b in blocks] or [_text(body)]
        parts.append(who + "\n" + "\n".join(lines))

    resolution = soup.select_one(".resolution")
    if resolution:
        parts.append("Resolution: " + _text(resolution))

    return ticket_id, title, fields, "\n\n".join(p for p in parts if p.strip())

def chunk_ticket(path: Path, corpus: Path) -> list[dict]:
    ticket_id, title, fields, text = clean_ticket(path.read_text(encoding="utf-8"))
    chunks = []
    for i, piece in enumerate(splitter.split_text(text)):
        # Later chunks get a breadcrumb so they still say which ticket they're from
        piece_text = piece if i == 0 else f"Ticket {ticket_id}: {title} (continued)\n\n{piece}"
        chunks.append({
            "text": piece_text,
            "source": rel_source(path, corpus),
            "doc_type": "tickets",
            "title": title,
            "ticket_id": ticket_id,
            "category": fields.get("category"),   # None values get dropped before upsert
            "status": fields.get("status"),
            "priority": fields.get("priority"),
        })
    return chunks

def with_ids(chunks: list[dict]) -> list[dict]:
    """Stable per-file IDs: 'source#0', 'source#1', ... plus a content hash."""
    for i, c in enumerate(chunks):
        c["id"] = f"{c['source']}#{i}"
        c["chunk_index"] = i
        c["content_hash"] = content_hash(c["text"])
    return chunks


def collect_chunks(corpus: Path) -> list[dict]:
    chunks = []
    for path in sorted((corpus / "product-docs").rglob("*.md")):
        chunks += with_ids(chunk_markdown(path, corpus))
    for path in sorted((corpus / "tickets").rglob("*.html")):
        chunks += with_ids(chunk_ticket(path, corpus))
    return chunks


def embed(oa: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        resp = oa.embeddings.create(model=EMBED_MODEL, input=texts[start:start + EMBED_BATCH])
        vectors += [item.embedding for item in resp.data]
    return vectors


def get_index(pc: Pinecone):
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    return pc.Index(INDEX_NAME)


def main():
    parser = argparse.ArgumentParser(description="Ingest the corpus into Pinecone.")
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--dry-run", action="store_true", help="chunk and print, no API calls")
    args = parser.parse_args()
    corpus = Path(args.corpus)

    chunks = collect_chunks(corpus)
    sizes = [n_tokens(c["text"]) for c in chunks]
    by_type = {t: sum(c["doc_type"] == t for c in chunks) for t in ("product-docs", "tickets")}
    print(f"{len(chunks)} chunks {by_type} | tokens min {min(sizes)} max {max(sizes)} total {sum(sizes):,}")

    if args.dry_run:
        samples = chunks[:2] + [c for c in chunks if c["doc_type"] == "tickets"][:2]
        for c in samples:
            print("-" * 70)
            print({k: v for k, v in c.items() if k != "text"})
            print(c["text"][:700])
        return

    oa, pc = OpenAI(), Pinecone()
    index = get_index(pc)
    if NAMESPACE in index.describe_index_stats().namespaces:
        index.delete(delete_all=True, namespace=NAMESPACE)   # full rebuild: no stale chunks

    vectors = embed(oa, [c["text"] for c in chunks])
    records = [
        {"id": c["id"], "values": vec,
         "metadata": {k: val for k, val in c.items() if k != "id" and val is not None}}
        for c, vec in zip(chunks, vectors)
    ]
    for start in range(0, len(records), UPSERT_BATCH):
        index.upsert(vectors=records[start:start + UPSERT_BATCH], namespace=NAMESPACE)
    print(f"Upserted {len(records)} chunks to {INDEX_NAME}/{NAMESPACE}")


if __name__ == "__main__":
    main()