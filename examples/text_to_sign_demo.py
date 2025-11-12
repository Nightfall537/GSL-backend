"""
Text-to-Sign Demo

Demonstrates how to use the Text-to-Sign model to convert text into GSL signs.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.text_to_sign import TextToSignModel


async def demo_basic_conversion():
    """Demo: Basic text to sign conversion."""
    print("=" * 60)
    print("DEMO 1: Basic Text-to-Sign Conversion")
    print("=" * 60)
    
    model = TextToSignModel()
    
    # Example texts
    texts = [
        "Hello, how are you?",
        "Thank you very much",
        "I need help",
        "Good morning",
        "What is your name?"
    ]
    
    for text in texts:
        print(f"\nText: '{text}'")
        signs = await model.convert_text_to_signs(text)
        
        print(f"Signs ({len(signs)}):")
        for i, sign in enumerate(signs, 1):
            print(f"  {i}. {sign['word']}")
            if sign.get('video_url'):
                print(f"     Video: {sign['video_url']}")
            if sign.get('description'):
                print(f"     Description: {sign['description']}")
        print("-" * 40)


async def demo_ghanaian_phrases():
    """Demo: Ghanaian phrase conversion."""
    print("\n" + "=" * 60)
    print("DEMO 2: Ghanaian Phrases")
    print("=" * 60)
    
    model = TextToSignModel()
    
    # Ghanaian phrases
    phrases = [
        "how far chale",
        "small small we go reach",
        "i beg help me",
        "by force you go do am"
    ]
    
    for phrase in phrases:
        print(f"\nGhanaian Phrase: '{phrase}'")
        signs = await model.convert_text_to_signs(phrase)
        
        print(f"Converted to signs:")
        for sign in signs:
            print(f"  → {sign['word']}")
        print("-" * 40)


async def demo_sign_sequence():
    """Demo: Generate complete sign sequence with timing."""
    print("\n" + "=" * 60)
    print("DEMO 3: Sign Sequence with Timing")
    print("=" * 60)
    
    model = TextToSignModel()
    
    text = "Hello, I want to learn sign language"
    print(f"\nText: '{text}'")
    
    sequence = await model.generate_sign_sequence(text)
    
    print(f"\nSign Sequence:")
    print(f"  Total Signs: {sequence['total_signs']}")
    print(f"  Estimated Duration: {sequence['estimated_duration']} seconds")
    print(f"\n  Signs:")
    for i, sign in enumerate(sequence['signs'], 1):
        print(f"    {i}. {sign['word']} ({sign.get('category', 'unknown')})")
    
    if sequence['transitions']:
        print(f"\n  Transitions:")
        for trans in sequence['transitions']:
            print(f"    {trans['from']} → {trans['to']} ({trans['duration']}s)")


async def demo_fingerspelling():
    """Demo: Fingerspelling for unknown words."""
    print("\n" + "=" * 60)
    print("DEMO 4: Fingerspelling")
    print("=" * 60)
    
    model = TextToSignModel()
    
    text = "My name is Kwame"
    print(f"\nText: '{text}'")
    
    # Without fingerspelling
    print("\nWithout fingerspelling:")
    signs = await model.convert_text_to_signs(text, include_fingerspelling=False)
    for sign in signs:
        print(f"  → {sign['word']}")
    
    # With fingerspelling
    print("\nWith fingerspelling:")
    signs = await model.convert_text_to_signs(text, include_fingerspelling=True)
    for sign in signs:
        sign_type = sign.get('type', 'sign')
        print(f"  → {sign['word']} [{sign_type}]")


async def demo_statistics():
    """Demo: Model statistics."""
    print("\n" + "=" * 60)
    print("DEMO 5: Model Statistics")
    print("=" * 60)
    
    model = TextToSignModel()
    
    stats = model.get_statistics()
    
    print(f"\nAvailable Signs: {stats['total_signs']}")
    print(f"Phrase Mappings: {stats['phrases']}")
    print(f"\nCategories:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count} signs")
    
    print(f"\nSample signs:")
    available_signs = model.get_available_signs()[:10]
    for sign in available_signs:
        print(f"  - {sign}")


async def demo_custom_signs():
    """Demo: Adding custom signs."""
    print("\n" + "=" * 60)
    print("DEMO 6: Custom Signs")
    print("=" * 60)
    
    model = TextToSignModel()
    
    # Add custom sign
    model.add_custom_sign(
        word="akwaaba",
        video_url="/videos/custom/akwaaba.mp4",
        description="Ghanaian welcome sign",
        category="greetings"
    )
    
    print("\nAdded custom sign: 'akwaaba'")
    
    # Use custom sign
    text = "akwaaba to Ghana"
    print(f"\nText: '{text}'")
    signs = await model.convert_text_to_signs(text)
    
    for sign in signs:
        print(f"  → {sign['word']}")
        if sign.get('custom'):
            print(f"     [CUSTOM SIGN]")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("TEXT-TO-SIGN MODEL DEMONSTRATION")
    print("Ghanaian Sign Language (GSL) Platform")
    print("=" * 60)
    
    try:
        await demo_basic_conversion()
        await demo_ghanaian_phrases()
        await demo_sign_sequence()
        await demo_fingerspelling()
        await demo_statistics()
        await demo_custom_signs()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())