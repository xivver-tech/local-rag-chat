# Local RAG Chat

Chat with your own documents using a local LLM via **Ollama**.

## Features
- Ingest a folder of `.txt` / `.md` files
- Chunks + simple but effective retrieval (no extra embedding server needed)
- Streams answers from Ollama
- Conversation history
- Fully local — nothing leaves your machine

## Requirements
1. Install [Ollama](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull llama3.2
   # or mistral, qwen2.5, phi3, etc.
   ```

## Usage
```bash
# 1. Index your documents
python rag.py ingest ./my-docs

# 2. Chat
python rag.py chat

# Use a different model
python rag.py chat mistral
```

Environment variables:
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3.2`)
