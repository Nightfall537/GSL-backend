# Gemma 2:2b Quick Start

## Installation (5 minutes)

```bash
# 1. Install Ollama
# Windows: Download from https://ollama.com/download
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve

# 3. Pull Gemma model (~1.6GB)
ollama pull gemma2:2b

# 4. Test it
ollama run gemma2:2b "Hello!"
```

## Basic Usage

```python
from app.ai.gemma_model import get_gemma_model

model = get_gemma_model()

# Generate text
response = await model.generate_async("What is sign language?")

# Chat
messages = [{"role": "user", "content": "Hello!"}]
response = await model.chat_async(messages)

# Stream
for chunk in model.stream_generate("Explain GSL"):
    print(chunk, end="")
```

## GSL-Specific Features

```python
# Simplify text for signing
simplified = await model.improve_translation(
    "I was wondering if you could help"
)
# → "Can you help me?"

# Explain signs simply
explanation = await model.explain_sign(
    "HELLO",
    "Wave hand down face with wiggling fingers"
)

# Generate practice sentences
sentences = await model.generate_practice_sentences(
    ["hello", "thank you"],
    count=5
)

# Translate Ghanaian phrases
result = await model.translate_ghanaian_phrase("how far")

# Generate lessons
lesson = await model.generate_lesson_content("Greetings", "beginner")
```

## Run Demo

```bash
python examples/gemma_model_demo.py
```

## Troubleshooting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List installed models
ollama list

# Reinstall model
ollama pull gemma2:2b
```

## Integration Example

```python
# Add to your API
from fastapi import APIRouter
from app.ai.gemma_model import get_gemma_model

router = APIRouter()

@router.post("/chat")
async def chat(message: str):
    model = get_gemma_model()
    response = await model.chat_async([
        {"role": "user", "content": message}
    ])
    return {"response": response}
```

## Key Features

✅ Local LLM (no API costs)
✅ Fast responses (~2-5 seconds)
✅ Async support
✅ Streaming
✅ GSL-specific helpers
✅ Conversation memory
✅ Temperature control

## Model Info

- **Size**: 1.6GB
- **Parameters**: 2 billion
- **Speed**: Fast on CPU
- **Context**: 8,192 tokens
- **Quality**: Excellent

---

**That's it! You're ready to use Gemma 2:2b in your GSL Backend! 🚀**