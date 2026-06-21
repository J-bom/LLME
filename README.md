<h1 align="center">LLME</h1>

![LLME](https://socialify.git.ci/J-bom/LLME/image?language=1&name=1&owner=1&pattern=Transparent&theme=Auto)

<p align="center"><b>A self-hosted, end-to-end-encrypted LLM platform - built from scratch.</b></p>

<p align="center">The ultimate AI frontend: an encrypted desktop client, a multi-backend model server, per-user document RAG, and a real Model Context Protocol (MCP) agent layer.</p>

---

## What it is

LLME is the serving and interface half of a two-part project. It loads and runs language models and gives users a private, secured chat experience over the network. It can serve three different kinds of model behind one interface:

- **AlterEgo** - a 373M-parameter transformer I trained from scratch ([code](https://github.com/J-bom/AlterEgo) · [weights](https://huggingface.co/jbomdev/AlterEgo))
- **GGUF models** via a `llama.cpp` backend (drop a model in and load it, like LM Studio)
- **Google Gemini** via API, for a long-context (up to 1M tokens) hosted option

The companion model project lives in the **AlterEgo** repo. This repo is the platform: client, server, networking, encryption, RAG, and tools.

## Architecture

<img src="assets/Architecture.png">

## Features

### 🔒 End-to-end encrypted transport
A hybrid scheme negotiated per connection: the client and server exchange **RSA-2048** public keys, the server generates a random **AES** session key and delivers it RSA-encrypted, and from then on every message is **AES-GCM** - authenticated encryption with a per-message nonce and integrity tag (`encrypt_and_digest` / `decrypt_and_verify`). A custom length-framed protocol carries 4-character message codes with JSON payloads and request/response IDs.

### 🧠 Multi-backend model serving
One server, three engines, dispatched by an engine-aware model loader with per-model locks and idle eviction:
| Backend | What it runs | Context |
|---|---|---|
| AlterEgo | My from-scratch 373M model (native PyTorch, KV-cache) | 2048 |
| llama.cpp | Any GGUF model you drop in | configurable |
| Gemini | Google's hosted models via API | up to 1,000,000 |

### 📚 Per-user document RAG
Upload documents and chat over them. Files are chunked, embedded with **BAAI/bge-base-en-v1.5** (Sentence-Transformers), and stored in a per-user vector store. Retrieval is exposed to the model as a callable tool (top-k semantic search), so the model pulls context on demand rather than stuffing every prompt. Uploads are bounded by a per-file size limit and a per-user storage quota (with in-flight reservations so concurrent uploads can't slip past the quota).

### 🛠️ MCP agent layer (function calling)
A real **Model Context Protocol** implementation, not an imitation. The client uses the official `mcp` SDK to spawn local MCP servers over stdio, lists their tools, and registers them with the server; the server keeps a per-user tool registry and offers the tools to the model for function calling. Ships with built-in server-side tools too: current time, an AST-sandboxed calculator (`safe_eval`), and RAG document search.

### 👤 Accounts, sessions & history
User accounts with per-user random salts, multi-session chat history persisted in SQLite, per-user model preferences, and resumable conversations.

### 🛡️ Auth-abuse & DoS protection
Login and signup are defended on two axes at once - failures are tracked per-username *and* per-source-IP over a sliding time window. Past a soft threshold the server applies an exponential backoff (escalating response delays), and past a hard threshold it locks the account or blocks the IP outright; signup gets its own per-IP rate limit to stop mass account creation. At the connection layer, the server caps concurrent connections per IP and drops connections that don't authenticate within a pre-auth timeout (anti-slowloris). The client surfaces these to the user with clear messages (e.g. account-locked, file-too-large).

### 🖥️ Desktop client
PySide6/Qt client with `.ui`-driven layouts, light/dark theme switching (QSS), streaming token output on a worker thread, and file attachment.

## Tech stack
Python · PySide6/Qt · sockets · PyCryptodome (RSA + AES-GCM) · SQLite · Sentence-Transformers · `llama-cpp-python` · `google-genai` · the official `mcp` SDK

## Repo layout
```
client/   PySide6 desktop app, local MCP host, transport
server/   model loader, RAG engine, tool registry/executor, auth, transport
```

## Running it

**Server**
```bash
cd server
pip install -r requirements.txt
# drop a GGUF model in Ai/Models/ and/or configure backends in config.py
python server.py
```

**Client**
```bash
cd client
pip install -r requirements.txt
# point local MCP servers at mcp_config.json (optional)
python client.py
```

## Credits
- Embeddings: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
- Local model runtime: [`llama.cpp`](https://github.com/ggerganov/llama.cpp) via `llama-cpp-python`
- Agent protocol: [Model Context Protocol](https://modelcontextprotocol.io)
- The AlterEgo model: [github.com/J-bom/AlterEgo](https://github.com/J-bom/AlterEgo)
