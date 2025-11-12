"""
Enhanced AI Models Demo

Demonstrates how all AI models can use Gemma for improved results.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.enhanced_models import (
    EnhancedNLPProcessor,
    EnhancedTranslationHelper,
    EnhancedLearningHelper,
    EnhancedConversationalAssistant,
    enhance_text_for_signing,
    explain_sign_simply,
    generate_lesson,
    chat_with_assistant
)


async def demo_enhanced_nlp():
    """Demo: Enhanced NLP with Gemma."""
    print("=" * 60)
    print("DEMO 1: Enhanced NLP Processing")
    print("=" * 60)
    
    nlp = EnhancedNLPProcessor(use_gemma=True)
    
    texts = [
        "I was wondering if you could possibly help me learn sign language",
        "How far chale, make we go learn small small",
        "What is the best way to practice signing every day?"
    ]
    
    for text in texts:
        print(f"\nOriginal: {text}")
        
        # Simplify
        simplified = await nlp.simplify_text(text)
        print(f"Simplified: {simplified}")
        
        # Extract keywords
        keywords = await nlp.extract_keywords_enhanced(text)
        print(f"Keywords: {', '.join(keywords[:5])}")
        
        # Detect intent
        intent = await nlp.detect_intent_enhanced(text)
        print(f"Intent: {intent['intent']} (confidence: {intent['confidence']})")
        print("-" * 40)


async def demo_translation_helper():
    """Demo: Translation helper."""
    print("\n" + "=" * 60)
    print("DEMO 2: Translation Helper")
    print("=" * 60)
    
    helper = EnhancedTranslationHelper()
    
    # Improve text
    text = "I would really appreciate it if you could help me"
    print(f"\nOriginal: {text}")
    improved = await helper.improve_for_signing(text)
    print(f"Improved: {improved}")
    
    # Explain sign
    print("\n" + "-" * 40)
    sign_name = "HELLO"
    description = "Place open hand near face, move down while wiggling fingers"
    print(f"\nSign: {sign_name}")
    print(f"Technical: {description}")
    explanation = await helper.explain_sign_simply(sign_name, description)
    print(f"Simple: {explanation}")
    
    # Generate examples
    print("\n" + "-" * 40)
    print(f"\nUsage examples for 'THANK YOU':")
    examples = await helper.generate_usage_examples("thank you", count=3)
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    # Suggest related signs
    print("\n" + "-" * 40)
    print(f"\nRelated signs to 'HELLO':")
    related = await helper.suggest_related_signs("hello", count=5)
    print(f"  {', '.join(related)}")


async def demo_learning_helper():
    """Demo: Learning content generation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Learning Helper")
    print("=" * 60)
    
    helper = EnhancedLearningHelper()
    
    # Generate lesson
    print("\nGenerating lesson...")
    lesson = await helper.generate_lesson_content(
        topic="Greetings",
        level="beginner",
        signs=["hello", "goodbye", "thank you"]
    )
    print(f"\nTopic: {lesson['topic']}")
    print(f"Level: {lesson['level']}")
    print(f"Content:\n{lesson['content'][:300]}...")
    
    # Generate dialogue
    print("\n" + "-" * 40)
    print("\nGenerating practice dialogue...")
    dialogue = await helper.generate_practice_dialogue(
        scenario="Meeting someone for the first time",
        signs_to_use=["hello", "name", "nice to meet you"]
    )
    print(f"\nDialogue:\n{dialogue[:300]}...")
    
    # Provide feedback
    print("\n" + "-" * 40)
    print("\nGenerating learning feedback...")
    feedback = await helper.provide_learning_feedback({
        "lessons_completed": 5,
        "accuracy": 85,
        "signs_learned": 25
    })
    print(f"\nFeedback: {feedback}")


async def demo_conversational_assistant():
    """Demo: Conversational assistant."""
    print("\n" + "=" * 60)
    print("DEMO 4: Conversational Assistant")
    print("=" * 60)
    
    assistant = EnhancedConversationalAssistant()
    
    # Conversation
    messages = [
        "Hello! I'm new to sign language.",
        "What should I learn first?",
        "How long does it take to become fluent?"
    ]
    
    for message in messages:
        print(f"\nUser: {message}")
        response = await assistant.chat(message)
        print(f"Assistant: {response}")
        print("-" * 40)
    
    # Answer specific question
    print("\n" + "=" * 40)
    question = "What's the difference between GSL and ASL?"
    print(f"\nQuestion: {question}")
    answer = await assistant.answer_question(question)
    print(f"Answer: {answer}")


async def demo_quick_functions():
    """Demo: Quick helper functions."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quick Helper Functions")
    print("=" * 60)
    
    # Enhance text
    print("\n1. Enhance text for signing:")
    text = "I was wondering if you could help"
    enhanced = await enhance_text_for_signing(text)
    print(f"   Original: {text}")
    print(f"   Enhanced: {enhanced}")
    
    # Explain sign
    print("\n2. Explain sign simply:")
    explanation = await explain_sign_simply(
        "WATER",
        "Tap chin with index finger three times"
    )
    print(f"   Explanation: {explanation}")
    
    # Generate lesson
    print("\n3. Generate lesson:")
    lesson = await generate_lesson("Family Members", "beginner")
    print(f"   Topic: {lesson['topic']}")
    print(f"   Content: {lesson['content'][:150]}...")
    
    # Chat
    print("\n4. Chat with assistant:")
    response = await chat_with_assistant("How do I practice signing at home?")
    print(f"   Response: {response[:150]}...")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ENHANCED AI MODELS DEMONSTRATION")
    print("All Models Using Gemma 2:2b via Ollama")
    print("=" * 60)
    
    print("\n⚠️  Make sure Ollama is running: ollama serve")
    print("⚠️  Make sure Gemma is installed: ollama pull gemma2:2b\n")
    
    try:
        await demo_enhanced_nlp()
        await demo_translation_helper()
        await demo_learning_helper()
        await demo_conversational_assistant()
        await demo_quick_functions()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nAll AI models now enhanced with Gemma 2:2b!")
        print("\nFeatures:")
        print("  ✓ Enhanced NLP processing")
        print("  ✓ Better text simplification")
        print("  ✓ Intelligent sign explanations")
        print("  ✓ Auto-generated lessons")
        print("  ✓ Practice dialogues")
        print("  ✓ Conversational assistant")
        print("  ✓ Learning feedback")
        
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