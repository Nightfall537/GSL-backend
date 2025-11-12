"""
Text-to-Speech Demo

Demonstrates how to use the Text-to-Speech model for audio synthesis.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.text_to_speech import TextToSpeechModel


async def demo_basic_synthesis():
    """Demo: Basic text-to-speech synthesis."""
    print("=" * 60)
    print("DEMO 1: Basic Text-to-Speech")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    texts = [
        "Hello, welcome to the GSL learning platform",
        "You are doing great!",
        "Let's practice some more signs",
        "Thank you for learning with us"
    ]
    
    for text in texts:
        print(f"\nText: '{text}'")
        
        # Synthesize audio
        audio_bytes = await model.synthesize(text)
        
        print(f"  Audio generated: {len(audio_bytes)} bytes")
        
        # Estimate duration
        duration = await model.estimate_duration(text)
        print(f"  Estimated duration: {duration:.2f} seconds")


async def demo_voice_options():
    """Demo: Different voice options."""
    print("\n" + "=" * 60)
    print("DEMO 2: Voice Options")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    # Get available voices
    voices = model.get_available_voices()
    
    print("\nAvailable Voices:")
    for voice in voices:
        print(f"  - {voice['name']} ({voice['id']})")
        print(f"    Language: {voice['language']}, Gender: {voice['gender']}")
    
    # Synthesize with different voices
    text = "Hello, how are you?"
    print(f"\nSynthesizing: '{text}'")
    
    for voice in voices[:2]:  # Try first 2 voices
        print(f"\n  Using voice: {voice['name']}")
        audio = await model.synthesize(text, voice=voice['id'])
        print(f"    Generated: {len(audio)} bytes")


async def demo_speed_and_pitch():
    """Demo: Speed and pitch control."""
    print("\n" + "=" * 60)
    print("DEMO 3: Speed and Pitch Control")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    text = "This is a test of speech synthesis"
    
    # Different speeds
    speeds = [0.75, 1.0, 1.25, 1.5]
    print(f"\nText: '{text}'")
    print("\nSpeed variations:")
    
    for speed in speeds:
        audio = await model.synthesize(text, speed=speed)
        duration = await model.estimate_duration(text, speed=speed)
        print(f"  Speed {speed}x: {duration:.2f}s, {len(audio)} bytes")
    
    # Different pitches
    pitches = [0.8, 1.0, 1.2]
    print("\nPitch variations:")
    
    for pitch in pitches:
        audio = await model.synthesize(text, pitch=pitch)
        print(f"  Pitch {pitch}x: {len(audio)} bytes")


async def demo_save_to_file():
    """Demo: Save audio to file."""
    print("\n" + "=" * 60)
    print("DEMO 4: Save Audio to File")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    text = "Welcome to Ghanaian Sign Language learning"
    output_path = "output_audio.wav"
    
    print(f"\nText: '{text}'")
    print(f"Output: {output_path}")
    
    success = await model.synthesize_to_file(text, output_path)
    
    if success:
        print("  ✓ Audio saved successfully")
        
        # Check file
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"  File size: {file_size} bytes")
    else:
        print("  ✗ Failed to save audio")


async def demo_duration_estimation():
    """Demo: Duration estimation for different texts."""
    print("\n" + "=" * 60)
    print("DEMO 5: Duration Estimation")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    texts = [
        "Hi",
        "Hello, how are you?",
        "Welcome to the Ghanaian Sign Language learning platform. We are excited to help you learn.",
        "This is a much longer text that will take more time to speak. It contains multiple sentences and should demonstrate how duration estimation works for longer passages."
    ]
    
    print("\nDuration estimates:")
    for text in texts:
        duration = await model.estimate_duration(text)
        word_count = len(text.split())
        print(f"\n  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Words: {word_count}")
        print(f"  Duration: {duration:.2f} seconds")


async def demo_ssml():
    """Demo: SSML (Speech Synthesis Markup Language)."""
    print("\n" + "=" * 60)
    print("DEMO 6: SSML Support")
    print("=" * 60)
    
    model = TextToSpeechModel()
    
    # SSML example
    ssml = """
    <speak>
        <prosody rate="slow">
            Hello
        </prosody>
        <break time="500ms"/>
        <prosody rate="fast">
            Welcome to GSL learning
        </prosody>
        <emphasis level="strong">
            Let's get started!
        </emphasis>
    </speak>
    """
    
    print("\nSSML Input:")
    print(ssml)
    
    audio = await model.synthesize_ssml(ssml)
    print(f"\nGenerated audio: {len(audio)} bytes")
    print("(SSML support is a placeholder - to be implemented)")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("TEXT-TO-SPEECH MODEL DEMONSTRATION")
    print("Audio Synthesis for GSL Platform")
    print("=" * 60)
    
    try:
        await demo_basic_synthesis()
        await demo_voice_options()
        await demo_speed_and_pitch()
        await demo_save_to_file()
        await demo_duration_estimation()
        await demo_ssml()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nNote: Audio synthesis uses placeholder implementation.")
        print("To use real TTS, integrate one of these libraries:")
        print("  - gTTS (Google Text-to-Speech)")
        print("  - pyttsx3 (Offline TTS)")
        print("  - Coqui TTS (High quality)")
        print("  - ElevenLabs API (Premium quality)")
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())