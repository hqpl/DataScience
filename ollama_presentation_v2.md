# How to manage LLM locally:
## 1. Ollama - https://ollama.com/
* What is Ollama
* How to install it - https://ollama.com/
* AMD hardware support - https://github.com/ollama/ollama/blob/main/docs/linux.md
* Managing models - https://ollama.com/library
    * run
	* show
	* ps
	* list
* Python Ollama library - https://pypi.org/project/ollama/
* Ollama frontend - https://github.com/ollama/ollama?tab=readme-ov-file#web--desktop
* Spring Framework AI Ollama Chat https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html
* Ollama in Docker - https://hub.docker.com/r/ollama/ollama
* Hugging Face models - https://huggingface.co/models?library=gguf&sort=trending

## 2. Llamma.cpp - https://github.com/ggerganov/llama.cpp
* How to install Llamma.cpp and run GGUF models from HF - https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#quick-start
* AMD hardware support - https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip
* Spring Framework AI Llamma.cpp Chat https://github.com/rsatrio/llm-chatbot-springboot

## 3. LM Studio - https://lmstudio.ai/
* How to install it - https://lmstudio.ai/download
* How to run it - https://lmstudio.ai/docs/app/basics
* AMD hardware support - Download ROCm llama.cpp inside LM StudioOptions -> Runtime, then select it as GGUF Engine

## 4. Docker - https://hub.docker.com/u/ai

## Connect to LLM via API calls:
* Ollama API - https://docs.ollama.com/api/introduction
* Llamma.cpp API - https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#a-lightweight-openai-api-compatible-http-server-for-serving-llms
* LM Studio API - https://lmstudio.ai/docs/api
* Docker API - https://docs.docker.com/engine/api/

# RAG
* ChromaDB - https://www.trychroma.com/
* Pinecone - https://www.pinecone.io/
* Milvus - https://milvus.io/
* Weaviate - https://weaviate.io/
* Qdrant - https://qdrant.tech/


## What's next?
* MCP - https://hub.docker.com/mcp
* ollmcp - https://github.com/jonigl/mcp-client-for-ollama
* UV - https://github.com/astral-sh/uv
* text to speach with llama.cpp or Torch - https://github.com/edwko/OuteTTS
* Fine Tuning with unsloth - https://docs.unsloth.ai/get-started/fine-tuning-llms-guide

