## How to manage LLMs locally with Ollama

### What is Ollama
- **Ollama** is a local LLM runtime and model manager for downloading, running, and serving open-source models with simple commands.
- Supports **GPU acceleration** on NVIDIA (CUDA) and AMD (ROCm); automatically falls back to CPU.
- Exposes a **REST API** and **Python library** for programmatic use.

### Install Ollama (1 command)
- Linux/macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
- Windows: download from https://ollama.com/ or use:
```powershell
winget install Ollama.Ollama
```

### Hardware acceleration (CUDA/ROCm)
- **NVIDIA (CUDA)**: install recent NVIDIA driver + CUDA runtime; Ollama auto-detects GPU.
- **AMD (ROCm)**: follow https://github.com/ollama/ollama/blob/main/docs/linux.md. Ollama uses ROCm when available.
- Verify usage during inference:
  - NVIDIA: `nvidia-smi` shows `ollama` using GPU memory.
  - AMD: `rocm-smi` shows active GPU utilization.
- Control via env vars:
```bash
# Force CPU
export OLLAMA_NO_GPU=1
# Limit number of GPU layers (varies by model)
export OLLAMA_NUM_GPU=20
```

### Finding and managing models
- Browse the **Ollama Library**: https://ollama.com/library
  - Popular: `llama3`, `qwen2.5`, `deepseek-r1` (thinking), `llava` (vision), `mistral`, `phi4`, `nomic-embed-text` (embeddings)

#### Core CLI commands
```bash
# Download a model
ollama pull llama3

# List downloaded models
ollama list

# Run interactive chat
ollama run llama3

# One-off prompt
ollama run llama3 "Write a haiku about local LLMs."

# Show model details (parameters, template, modelfile)
ollama show llama3

# Show running sessions, shows whether a running model is on CPU or GPU.
ollama ps

# Start/stop server (usually auto-managed)
ollama serve
pkill ollama
```

#### Is it using CPU or GPU?
- Run a model, then:
  - NVIDIA: `nvidia-smi` should list an `ollama` process with GPU memory usage.
  - AMD: `rocm-smi` should show usage.
- Server startup logs (when running `ollama serve`) often indicate CUDA/ROCm detection.

### Python: Ollama on PyPI
- PyPI: https://pypi.org/project/ollama/
```bash
pip install ollama
```

#### Chat (conversational text generation)
- Use when you want the model to generate natural language responses in a multi-turn conversation, code assistance, drafting, etc.
- Input is a list of messages with roles (e.g., `user`, `assistant`, `system`), output is the model’s next message.
```python
import ollama

resp = ollama.chat(
    model="llama3",
    messages=[
        {"role": "user", "content": "Summarize Ollama in 3 bullets."}
    ],
)
print(resp["message"]["content"])
```

#### Embeddings (vector representations for search and RAG)
- Use when you need a numeric vector that captures semantic meaning of text for similarity search, clustering, retrieval-augmented generation (RAG), and recommendations.
- Output is a dense vector (list of floats) suitable for storage in a vector database (e.g., FAISS, Chroma, pgvector).
```python
import ollama

emb = ollama.embeddings(
    model="nomic-embed-text",
    prompt="Local models are great for privacy."
)
print(len(emb["embedding"]))  # e.g., 768 or 1024 dims depending on the model
```
- The PyPI page documents `chat`, `generate`, `embeddings`, model management, and server options.

### Ollama Library tags explained
- **tools**: supports function calling (tool use) with structured outputs to call external APIs.
- **thinking**: enhanced long-form reasoning traces and multi-step problem solving (e.g., DeepSeek-R1).
- **vision**: multimodal—accepts and reasons over images (e.g., LLaVA, Qwen-VL).
- **cloud**: integrates with or requires remote/cloud-backed functionality.
- Other common tags: **code**, **math**, **embed** (embeddings), **instruct** (chat-tuned).

### Run Ollama in Docker (alternative to local install)
- Docker Hub: https://hub.docker.com/r/ollama/ollama
```bash
# CPU or generic
docker run -d --name ollama -p 11434:11434 \
  -v ollama:/root/.ollama ollama/ollama

# NVIDIA GPU
docker run -d --gpus all --name ollama -p 11434:11434 \
  -v ollama:/root/.ollama ollama/ollama

# Pull and run inside the container
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama run llama3
```

### Hugging Face vs Ollama/LM Studio
- **Hugging Face Models**: https://huggingface.co/models?library=gguf&sort=trending
  - Enormous catalog: LLMs, embeddings, **text-to-speech**, **text-to-image**, **text-to-video**, ASR, diffusion, etc.
  - Many available in **GGUF** (works with llama.cpp/Ollama/LM Studio). Others can be converted or used via framework-native runtimes.

### Alternatives and comparison
- **Ollama**
  - Pros: simple CLI/API, quick setup, CUDA/ROCm support, curated library, easy Docker.
  - Cons: focused on text/chat/embeddings; complex pipelines require composition with other tools.
- **LM Studio**
  - Pros: polished desktop UI, built-in model browser, good GGUF support, local server mode.
  - Cons: desktop-centric; automation is less direct than Ollama’s CLI/API.
- **Claude Desktop**
  - Pros: excellent UX for Anthropic Claude; integrates with local files.
  - Cons: not a local open-source runtime; relies on cloud models; limited to Claude family.
- **Docker (generic model containers)**
  - Pros: isolation, reproducibility, portable deployments, vendor images (e.g., https://hub.docker.com/u/ai).
  - Cons: GPU passthrough and setup overhead; fragmented interfaces vs Ollama’s unified CLI.

### Useful extras
- Spring AI integration (Java): https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html
- Next steps:
  - **MCP** (Model Context Protocol) clients/servers
  - **UV** for Python environments
  - **ollmcp**: https://github.com/jonigl/mcp-client-for-ollama
  - **Text-to-speech**: https://github.com/edwko/OuteTTS or llama.cpp TTS
  - **Databases/RAG**: vector DBs (FAISS, Chroma) with Ollama embeddings
  - **Fine-tuning**: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide


