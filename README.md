# Local RAG Service With NiceGui, Qdrant, Phi4-Mini, and MLflow Tracing

FastAPI service for local document RAG:

1. Parse local `.pdf`, `.md`, and `.txt` files.
2. Chunk and embed them locally with FastEmbed.
3. Store chunks in Qdrant.
4. Answer questions with local Ollama `phi4-mini`.
5. Trace ingestion, retrieval, prompt assembly, and chat calls in MLflow.

## Setup

Python Version is 3.12.0

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

- `QDRANT_API_KEY`
- `RAG_DOCS_ROOT`
- `RAG_MANIFEST_PATH=.rag_manifest.json`
- `FASTEMBED_MODEL=mixedbread-ai/mxbai-embed-large-v1`
- `FASTEMBED_VECTOR_SIZE=1024`
- `FASTEMBED_CUDA=true`
- `FASTEMBED_PROVIDERS=CUDAExecutionProvider`
- `FASTEMBED_DEVICE_IDS=0`

The app reads `.env` automatically. Real environment variables still override values in
that file.

The default MLflow tracking URI is:

```text
http://mlflow.tail.....net:5000/
```

When MLflow is reachable, the app loads the system prompt from the MLflow Prompt
Registry using `MLFLOW_PROMPT_NAME` and `MLFLOW_PROMPT_ALIAS`. If the prompt does
not exist yet, the app registers the default balanced RAG prompt and assigns the
alias. Chat traces include the browser/API session id, configured user name,
model, prompt version, retrieval settings, and returned source filenames.

The default embedding model is `mixedbread-ai/mxbai-embed-large-v1`, which
uses 1024-dimensional vectors. `/ingest` with `"reset": false` skips source
files when the local manifest shows the same file hash, Qdrant collection,
embedding model, vector size, and chunk settings. Use `"reset": true` when you
intentionally want to rebuild the collection.

The manifest is local app state. If you modify Qdrant outside this app, run
`/ingest` with `"reset": true` to rebuild the collection and refresh the
manifest.

GPU embeddings are enabled by default. Install the GPU build in the same Python
environment used to run Uvicorn:

```bash
python -m pip uninstall -y fastembed onnxruntime
python -m pip install -r requirements.txt
```

The app also preloads CUDA/cuDNN libraries installed from the NVIDIA Python
wheels before FastEmbed initializes ONNX Runtime. This prevents silent CPU
fallback when GPU embeddings are enabled.

Runtime timing logs are emitted for MLflow setup, embedding, retrieval, Qdrant
upserts/searches, and Ollama generation. They are useful for identifying where
slow `/chat` requests are spending time.

Verify CUDA is available before ingesting:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

The output must include `CUDAExecutionProvider`.

## Ollama

The planned chat model is `phi4-mini`. If your Ollama version is too old for that model, upgrade Ollama first, then pull it:

```bash
ollama --version
ollama pull <model-name>
ollama serve # (serves model so we can hit the endpoint)
```

## Run

# Starts the server

```bash
PYTHONPATH=src python -m uvicorn rag_app.main:app --host 0.0.0.0 --port 8001
```

Browser chat:

```text
http://127.0.0.1:8001/
```

The browser UI is built with NiceGUI. It keeps the existing JSON API routes and
adds a document panel for uploading `.pdf`, `.md`, `.txt`, and `.zip` files.
Uploaded documents are saved under `RAG_DOCS_ROOT/uploads/`; zip archives are
extracted under `RAG_DOCS_ROOT/imports/`. Both paths are indexed with the same
ingest flow used by `/ingest`.

Health:

```bash
curl http://localhost:8001/health
```

## Tests

```bash
pytest
```

Live service tests are intentionally not run by default. Use `RUN_LIVE_TESTS=1` only after configuring Qdrant, Ollama, and MLflow.
