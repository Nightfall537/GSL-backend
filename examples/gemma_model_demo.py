"""
Gemma 2:2b Model Demo

Demonstrates how to use the Gemma model for various GSL-related tasks.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.gemma_model import GemmaModel


async def demo_basic_generation():
    """Demo: Basic text generation."""
    print("=" * 60)
    print("DEMO 1: Basic Text Generation")
    print("=" * 60)
    
    model = GemmaModel()
    
    prompts = [
        "What is sign language?",
        "Explain the importance of learning sign language in Ghana",
        "List 5 benefits of learning Ghanaian Sign Language"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = await model.generate_async(prompt, temperature=0.7)
        print(f"Response: {response[:200]}...")
        print("-" * 40)


async def demo_chat_conversation():
    """Demo: Chat conversation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Chat Conversation")
    print("=" * 60)
    
    model = GemmaModel()
    
    # Conversation history
    messages = [
        {"role": "user", "content": "Hello! I want to learn Ghanaian Sign Language."},
    ]
    
    print("\nUser: Hello! I want to learn Ghanaian Sign Language.")
    
    # Get response
    response = await model.chat_async(messages, temperature=0.7)
    print(f"Gemma: {response}")
    
    # Continue conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What should I learn first?"})
    
    print("\nUser: What should I learn first?")
    response = await model.chat_async(messages, temperature=0.7)
    print(f"Gemma: {response}")


async def demo_improve_translation():
    """Demo: Improve text for sign translation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Improve Text for Sign Translation")
    print("=" * 60)
    
    model = GemmaModel()
    
    texts = [
        "I was wondering if you could possibly help me with something",
        "The weather today is absolutely magnificent and wonderful",
        "Could you perhaps maybe assist me in finding the location?"
    ]
    
    for text in texts:
        print(f"\nOriginal: {text}")
        improved = await model.improve_translation(text)
        print(f"Simplified: {improved}")
        print("-" * 40)


async def demo_explain_sign():
    """Demo: Explain signs in simple terms."""
    print("\n" + "=" * 60)
    print("DEMO 4: Explain Signs")
    print("=" * 60)
    
    model = GemmaModel()
    
    signs = [
        {
            "name": "HELLO",
            "description": "Place the open hand near the top of your face with the palm facing you. Move the hand straight down the length of your face while wiggling or fluttering your fingers."
        },
        {
            "name": "THANK YOU",
            "description": "Touch your chin with your fingertips, then move your hand forward and down."
        }
    ]
    
    for sign in signs:
        print(f"\nSign: {sign['name']}")
        print(f"Technical: {sign['description']}")
        
        explanation = await model.explain_sign(sign['name'], sign['description'])
        print(f"Simple explanation: {explanation}")
        print("-" * 40)


async def demo_practice_sentences():
    """Demo: Generate practice sentences."""
    print("\n" + "=" * 60)
    print("DEMO 5: Generate Practice Sentences")
    print("=" * 60)
    
    model = GemmaModel()
    
    sign_sets = [
        ["hello", "thank you", "please"],
        ["family", "mother", "father"],
        ["food", "water", "eat"]
    ]
    
    for signs in sign_sets:
        print(f"\nSigns: {', '.join(signs)}")
        sentences = await model.generate_practice_sentences(signs, count=3)
        
        print("Practice sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"  {i}. {sentence}")
        print("-" * 40)


async def demo_ghanaian_phrases():
    """Demo: Translate Ghanaian phrases."""
    print("\n" + "=" * 60)
    print("DEMO 6: Translate Ghanaian Phrases")
    print("=" * 60)
    
    model = GemmaModel()
    
    phrases = [
        "how far",
        "chale",
        "small small",
        "by force",
        "i beg"
    ]
    
    for phrase in phrases:
        print(f"\nPhrase: '{phrase}'")
        result = await model.translate_ghanaian_phrase(phrase)
        print(f"Translation: {result['translation']}")
        print("-" * 40)


async def demo_lesson_generation():
    """Demo: Generate lesson content."""
    print("\n" + "=" * 60)
    print("DEMO 7: Generate Lesson Content")
    print("=" * 60)
    
    model = GemmaModel()
    
    topics = [
        ("Greetings", "beginner"),
        ("Family Members", "beginner"),
        ("Daily Activities", "intermediate")
    ]
    
    for topic, level in topics:
        print(f"\nTopic: {topic} ({level})")
        lesson = await model.generate_lesson_content(topic, level)
        print(f"Lesson:\n{lesson[:300]}...")
        print("-" * 40)


async def demo_streaming():
    """Demo: Streaming text generation."""
    print("\n" + "=" * 60)
    print("DEMO 8: Streaming Generation")
    print("=" * 60)
    
    model = GemmaModel()
    
    prompt = "Explain the benefits of learning sign language in 3 points"
    
    print(f"\nPrompt: {prompt}")
    print("Response (streaming): ", end="", flush=True)
    
    for chunk in model.stream_generate(prompt, temperature=0.7):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 40)


async def demo_model_info():
    """Demo: Get model information."""
    print("\n" + "=" * 60)
    print("DEMO 9: Model Information")
    print("=" * 60)
    
    model = GemmaModel()
    
    info = model.get_model_info()
    
    if info:
        print(f"\nModel: {model.model_name}")
        print(f"Details: {info.get('modelfile', 'N/A')[:200]}...")
    else:
        print("\nCould not retrieve model information")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("GEMMA 2:2B MODEL DEMONSTRATION")
    print("Local LLM Integration for GSL Platform")
    print("=" * 60)
    
    print("\n⚠️  Make sure Ollama is running: ollama serve")
    print("⚠️  Make sure Gemma is installed: ollama pull gemma2:2b\n")
    
    try:
        await demo_basic_generation()
        await demo_chat_conversation()
        await demo_improve_translation()
        await demo_explain_sign()
        await demo_practice_sentences()
        await demo_ghanaian_phrases()
        await demo_lesson_generation()
        await demo_streaming()
        await demo_model_info()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nGemma 2:2b is ready for:")
        print("  ✓ Text generation and completion")
        print("  ✓ Conversational AI")
        print("  ✓ Translation assistance")
        print("  ✓ Lesson content generation")
        print("  ✓ Sign explanations")
        print("  ✓ Practice sentence generation")
        
    except Exception as e:
        print(f"\n✗ Error running demo: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if Ollama is running: ollama serve")
        print("  2. Check if Gemma is installed: ollama list")
        print("  3. Install Gemma if needed: ollama pull gemma2:2b")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())