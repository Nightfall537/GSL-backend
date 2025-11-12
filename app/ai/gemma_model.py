"""
Gemma 2:2b Model Integration

Integrates Google's Gemma 2:2b model via Ollama for local LLM capabilities.
Useful for text generation, translation assistance, conversation, and NLP tasks.
"""

import json
import requests
from typing import Optional, List, Dict, Any, Generator
import asyncio
import aiohttp


class GemmaModel:
    """Wrapper for Gemma 2:2b model via Ollama."""
    
    def __init__(
        self,
        model_name: str = "gemma2:2b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Gemma model.
        
        Args:
            model_name: Ollama model name (default: gemma2:2b)
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self._verify_model()
    
    def _verify_model(self) -> None:
        """Verify that the model is available in Ollama."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if self.model_name not in model_names:
                    print(f"⚠️  Model '{self.model_name}' not found in Ollama")
                    print(f"   Available models: {', '.join(model_names)}")
                    print(f"   Run: ollama pull {self.model_name}")
                else:
                    print(f"✓ Gemma model '{self.model_name}' is ready")
        except Exception as e:
            print(f"⚠️  Could not connect to Ollama: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using Gemma model.
        
        Args:
            prompt: Input prompt
            system: System message/context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        url = f"{self.api_url}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Async version of generate.
        
        Args:
            prompt: Input prompt
            system: System message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        url = f"{self.api_url}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        print(f"Error: {response.status}")
                        return ""
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Chat with Gemma model using conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Model response
        """
        url = f"{self.api_url}/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
    
    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """
        Async version of chat.
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            
        Returns:
            Model response
        """
        url = f"{self.api_url}/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message", {}).get("content", "")
                    else:
                        return ""
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
    
    def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Stream text generation token by token.
        
        Args:
            prompt: Input prompt
            system: System message
            temperature: Sampling temperature
            
        Yields:
            Generated text chunks
        """
        url = f"{self.api_url}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error streaming: {e}")
            yield ""
    
    def _handle_stream(self, response) -> str:
        """Handle streaming response."""
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    full_response += chunk
                except json.JSONDecodeError:
                    continue
        
        return full_response
    
    # GSL-specific helper methods
    
    async def improve_translation(
        self,
        text: str,
        context: str = "Ghanaian Sign Language"
    ) -> str:
        """
        Improve text for better sign language translation.
        
        Args:
            text: Input text
            context: Translation context
            
        Returns:
            Improved text
        """
        prompt = f"""Simplify the following text for {context} translation.
Make it clear, concise, and easy to translate into sign language.
Remove unnecessary words and complex grammar.

Original text: {text}

Simplified text:"""
        
        return await self.generate_async(prompt, temperature=0.3)
    
    async def explain_sign(
        self,
        sign_name: str,
        sign_description: str
    ) -> str:
        """
        Generate a clear explanation of a sign.
        
        Args:
            sign_name: Name of the sign
            sign_description: Technical description
            
        Returns:
            User-friendly explanation
        """
        prompt = f"""Explain this Ghanaian Sign Language sign in simple, clear terms:

Sign: {sign_name}
Technical description: {sign_description}

Provide a simple explanation that a beginner can understand:"""
        
        return await self.generate_async(prompt, temperature=0.5)
    
    async def generate_practice_sentences(
        self,
        signs: List[str],
        count: int = 5
    ) -> List[str]:
        """
        Generate practice sentences using specific signs.
        
        Args:
            signs: List of sign words
            count: Number of sentences to generate
            
        Returns:
            List of practice sentences
        """
        signs_str = ", ".join(signs)
        prompt = f"""Generate {count} simple practice sentences that use these Ghanaian Sign Language signs: {signs_str}

Make the sentences:
- Simple and clear
- Useful for daily communication
- Appropriate for beginners

Sentences:"""
        
        response = await self.generate_async(prompt, temperature=0.7)
        
        # Parse sentences
        sentences = [s.strip() for s in response.split('\n') if s.strip()]
        return sentences[:count]
    
    async def translate_ghanaian_phrase(
        self,
        phrase: str
    ) -> Dict[str, str]:
        """
        Translate Ghanaian English phrase to standard English.
        
        Args:
            phrase: Ghanaian English phrase
            
        Returns:
            Dictionary with translation and explanation
        """
        prompt = f"""Translate this Ghanaian English phrase to standard English and explain its meaning:

Phrase: "{phrase}"

Provide:
1. Standard English translation
2. Brief explanation of the phrase

Response:"""
        
        response = await self.generate_async(prompt, temperature=0.3)
        
        return {
            "original": phrase,
            "translation": response,
            "context": "Ghanaian English"
        }
    
    async def generate_lesson_content(
        self,
        topic: str,
        level: str = "beginner"
    ) -> str:
        """
        Generate lesson content for GSL learning.
        
        Args:
            topic: Lesson topic
            level: Difficulty level
            
        Returns:
            Lesson content
        """
        prompt = f"""Create a {level} level lesson for learning Ghanaian Sign Language on the topic: {topic}

Include:
1. Introduction
2. Key signs to learn
3. Practice exercises
4. Tips for remembering

Lesson:"""
        
        return await self.generate_async(prompt, temperature=0.7, max_tokens=500)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": self.model_name},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {}


# Singleton instance
_gemma_model = None


def get_gemma_model() -> GemmaModel:
    """Get singleton instance of GemmaModel."""
    global _gemma_model
    if _gemma_model is None:
        _gemma_model = GemmaModel()
    return _gemma_model