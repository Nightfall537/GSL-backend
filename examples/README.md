# GSL Backend Examples

Example scripts demonstrating how to use the AI models in the GSL Backend.

## Available Examples

### 1. Text-to-Sign Demo (`text_to_sign_demo.py`)

Demonstrates converting text into GSL (Ghanaian Sign Language) signs.

**Features:**
- Basic text-to-sign conversion
- Ghanaian phrase handling
- Sign sequence generation with timing
- Fingerspelling for unknown words
- Model statistics
- Custom sign addition

**Run:**
```bash
python examples/text_to_sign_demo.py
```

**Example Output:**
```
Text: 'Hello, how are you?'
Signs (3):
  1. hello
     Video: /videos/signs/hello.mp4
  2. how
     Video: /videos/signs/how.mp4
  3. you
     Video: /videos/signs/you.mp4
```

### 2. Text-to-Speech Demo (`text_to_speech_demo.py`)

Demonstrates converting text to spoken audio.

**Features:**
- Basic text-to-speech synthesis
- Multiple voice options (Ghanaian English, US English)
- Speed and pitch control
- Save audio to file
- Duration estimation
- SSML support (placeholder)

**Run:**
```bash
python examples/text_to_speech_demo.py
```

**Example Output:**
```
Text: 'Hello, welcome to the GSL learning platform'
  Audio generated: 1024 bytes
  Estimated duration: 3.20 seconds
```

## Use Cases

### Text-to-Sign Model

**1. Learning Platform**
```python
from app.ai.text_to_sign import get_text_to_sign_model

model = get_text_to_sign_model()

# Convert user input to signs
text = "I want to learn"
signs = await model.convert_text_to_signs(text)

# Display sign videos to user
for sign in signs:
    show_video(sign['video_url'])
```

**2. Translation Service**
```python
# Translate Ghanaian phrases
text = "how far chale"
signs = await model.convert_text_to_signs(text)
# Returns: ["hello", "how", "you", "friend"]
```

**3. Practice Mode**
```python
# Generate practice sequence
sequence = await model.generate_sign_sequence(
    "Good morning teacher",
    grammar_rules=True
)

# Show signs with timing
for sign in sequence['signs']:
    display_sign(sign, duration=2.0)
```

### Text-to-Speech Model

**1. Audio Feedback**
```python
from app.ai.text_to_speech import get_tts_model

model = get_tts_model()

# Provide audio feedback
feedback = "Great job! You signed 'hello' correctly."
audio = await model.synthesize(feedback, voice="en-gh-female")
play_audio(audio)
```

**2. Lesson Narration**
```python
# Narrate lesson content
lesson_text = "In this lesson, we will learn basic greetings"
audio = await model.synthesize(lesson_text, speed=0.9)
save_audio(audio, "lesson_intro.wav")
```

**3. Pronunciation Guide**
```python
# Help with pronunciation
word = "akwaaba"
audio = await model.synthesize(
    word,
    voice="en-gh-male",
    speed=0.75  # Slower for learning
)
```

## Integration Examples

### API Endpoint Integration

```python
# app/api/v1/translate.py
from fastapi import APIRouter
from app.ai.text_to_sign import get_text_to_sign_model

router = APIRouter()

@router.post("/text-to-sign")
async def convert_text_to_sign(text: str):
    """Convert text to GSL signs."""
    model = get_text_to_sign_model()
    signs = await model.convert_text_to_signs(text)
    
    return {
        "text": text,
        "signs": signs,
        "count": len(signs)
    }
```

### Service Integration

```python
# app/services/translation_service.py
from app.ai.text_to_sign import get_text_to_sign_model

class TranslationService:
    def __init__(self):
        self.text_to_sign = get_text_to_sign_model()
    
    async def translate_text(self, text: str):
        signs = await self.text_to_sign.convert_text_to_signs(text)
        return self._format_response(signs)
```

## Model Configuration

### Text-to-Sign Configuration

```python
# Custom signs data path
model = TextToSignModel(
    signs_data_path="path/to/custom_signs.json"
)

# Add custom signs
model.add_custom_sign(
    word="akwaaba",
    video_url="/videos/akwaaba.mp4",
    description="Ghanaian welcome",
    category="greetings"
)
```

### Text-to-Speech Configuration

```python
# Configure voice and settings
model = TextToSpeechModel()

# Synthesize with options
audio = await model.synthesize(
    text="Hello",
    voice="en-gh-female",
    speed=1.0,
    pitch=1.0
)
```

## Data Format

### Signs Data JSON Format

```json
{
  "signs": [
    {
      "word": "hello",
      "video_url": "/videos/signs/hello.mp4",
      "thumbnail_url": "/videos/thumbs/hello.jpg",
      "description": "Greeting sign",
      "category": "greetings",
      "difficulty": 1,
      "duration": 2.0
    }
  ]
}
```

## Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# For TTS (optional)
pip install gtts  # Google TTS
pip install pyttsx3  # Offline TTS
pip install TTS  # Coqui TTS
```

## Testing

```bash
# Run text-to-sign demo
python examples/text_to_sign_demo.py

# Run text-to-speech demo
python examples/text_to_speech_demo.py

# Run with custom data
python examples/text_to_sign_demo.py --data custom_signs.json
```

## Next Steps

1. **Add Real TTS Library**: Replace placeholder with actual TTS implementation
2. **Train Custom Models**: Fine-tune models for Ghanaian English
3. **Add More Signs**: Expand the GSL dictionary
4. **Improve Grammar**: Implement GSL-specific grammar rules
5. **Add Animations**: Generate sign animations for missing videos

## Resources

- **GSL Dictionary**: `app/models/colors_signs_data.json`
- **AI Models**: `app/ai/`
- **Services**: `app/services/translation_service.py`
- **API Endpoints**: `app/api/v1/translate.py`

## Support

For questions or issues:
- Check the main README.md
- Review the API documentation at `/docs`
- See SUPABASE_INTEGRATION.md for database setup