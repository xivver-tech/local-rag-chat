#!/usr/bin/env python3
"""
Local RAG Chat
--------------
Chat with your local documents using Ollama.

Requirements:
  - Ollama running (https://ollama.com)
  - A model pulled, e.g.  ollama pull llama3.2

How it works:
  1. Ingest a folder of .txt / .md files → chunk + index
  2. On each question: find relevant chunks (keyword + overlap scoring)
  3. Send context + question to Ollama
  4. Stream the answer

No external embedding server required (uses simple but effective retrieval).
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from collections import Counter

INDEX_FILE = Path(__file__).parent / "index.json"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ---------- chunking ----------

def chunk_text(text, size=500, overlap=80):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks

def ingest(folder):
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return

    docs = []
    for path in folder.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md", ".markdown", ".rst"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": f"{path.name}::{i}",
                "source": str(path.relative_to(folder)),
                "text": chunk
            })
        print(f"  {path.name}: {len(chunk_text(text))} chunks")

    INDEX_FILE.write_text(json.dumps({"docs": docs}, indent=2), encoding="utf-8")
    print(f"\n✓ Indexed {len(docs)} chunks from {folder}")

def load_index():
    if not INDEX_FILE.exists():
        print("No index found. Run: python rag.py ingest <folder>")
        sys.exit(1)
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))["docs"]

# ---------- retrieval ----------

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def score_chunk(query_tokens, chunk_text):
    chunk_tokens = tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0
    q_counts = Counter(query_tokens)
    c_counts = Counter(chunk_tokens)
    # simple TF overlap score
    score = sum(min(q_counts[t], c_counts[t]) for t in q_counts)
    # boost if query terms appear close together
    return score / (1 + len(chunk_tokens) ** 0.3)

def retrieve(docs, query, k=5):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for doc in docs:
        s = score_chunk(q_tokens, doc["text"])
        if s > 0:
            scored.append((s, doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]

# ---------- ollama ----------

def ollama_chat(messages, model=DEFAULT_MODEL, stream=True):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if not stream:
                data = json.loads(resp.read().decode())
                return data.get("message", {}).get("content", "")

            full = []
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full.append(token)
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            print()
            return "".join(full)
    except urllib.error.URLError as e:
        print(f"\nError talking to Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running:  ollama serve")
        print("And you have a model:        ollama pull", DEFAULT_MODEL)
        print("Details:", e)
        sys.exit(1)

# ---------- chat loop ----------

def chat(model=DEFAULT_MODEL):
    docs = load_index()
    print(f"Loaded {len(docs)} chunks. Model: {model}")
    print("Type your question (or 'quit')\n")

    system = (
        "You are a helpful assistant. Answer the user using ONLY the provided context. "
        "If the context is insufficient, say you don't know based on the documents. "
        "Be concise and cite the source filenames when possible."
    )

    history = [{"role": "system", "content": system}]

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break

        hits = retrieve(docs, q, k=5)
        if not hits:
            context = "(No relevant documents found)"
        else:
            parts = []
            for h in hits:
                parts.append(f"[{h['source']}]\n{h['text']}")
            context = "\n\n---\n\n".join(parts)

        user_msg = f"Context:\n{context}\n\nQuestion: {q}"
        history.append({"role": "user", "content": user_msg})

        print("\nAssistant: ", end="", flush=True)
        answer = ollama_chat(history, model=model, stream=True)
        history.append({"role": "assistant", "content": answer})

        # keep history from growing too large
        if len(history) > 13:  # system + 6 turns
            history = [history[0]] + history[-12:]

def main():
    if len(sys.argv) < 2:
        print("""Local RAG Chat (Ollama)
=======================
  ingest <folder>     Index all .txt/.md files in a folder
  chat [model]        Start chatting with your documents

Examples:
  python rag.py ingest ./docs
  python rag.py chat
  python rag.py chat mistral

Environment:
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default llama3.2
""")
        return

    cmd = sys.argv[1].lower()
    if cmd == "ingest" and len(sys.argv) > 2:
        ingest(sys.argv[2])
    elif cmd == "chat":
        model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
        chat(model)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
