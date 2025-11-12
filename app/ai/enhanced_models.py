"""
Enhanced AI Models with Gemma Integration

Provides enhanced versions of AI models that use Gemma for improved results.
All models can optionally use the local Gemma 2:2b model via Ollama.
"""

from typing import Optional, List, Dict
from app.ai.gemma_model import get_gemma_model, GemmaModel
from app.ai.nlp_processor import NLPProcessor


class EnhancedNLPProcessor:
    """NLP Processor enhanced with Gemma for better understanding."""
    
    def __init__(self, use_gemma: bool = True):
        """
        Initialize enhanced NLP processor.
        
        Args:
            use_gemma: Whether to use Gemma for enhancement
        """
        self.base_nlp = NLPProcessor()
        self.use_gemma = use_gemma
        self.gemma = get_gemma_model() if use_gemma else None
    
    async def extract_keywords_enhanced(self, text: str) -> List[str]:
        """
        Extract keywords with Gemma enhancement.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Get base keywords
        base_keywords = await self.base_nlp.extract_keywords(text)
        
        if not self.use_gemma or not self.gemma:
            return base_keywords
        
        # Enhance with Gemma
        prompt = f"""Extract the most important keywords from this text for sign language translation:

Text: {text}

List only the key words (comma-separated):"""
        
        response = await self.gemma.generate_async(prompt, temperature=0.3)
        
        # Parse Gemma's keywords
        gemma_keywords = [k.strip().lower() for k in response.split(',')]
        
        # Combine and deduplicate
        all_keywords = list(set(base_keywords + gemma_keywords))
        
        return all_keywords
    
    async def simplify_text(self, text: str) -> str:
        """
        Simplify text for sign language translation using Gemma.
        
        Args:
            text: Input text
            
        Returns:
            Simplified text
        """
        if not self.use_gemma or not self.gemma:
            return await self.base_nlp.process_text(text)
        
        return await self.gemma.improve_translation(text)
    
    async def detect_intent_enhanced(self, text: str) -> Dict[str, any]:
        """
        Detect intent with confidence scores using Gemma.
        
        Args:
            text: Input text
            
        Returns:
            Intent information with confidence
        """
        # Get base intent
        base_intent = self.base_nlp.detect_intent(text)
        
        if not self.use_gemma or not self.gemma:
            return {"intent": base_intent, "confidence": 0.7}
        
        # Enhance with Gemma
        prompt = f"""Analyze the intent of this text:

Text: "{text}"

What is the primary intent? Choose one: question, greeting, request, statement, command

Intent:"""
        
        response = await self.gemma.generate_async(prompt, temperature=0.2)
        gemma_intent = response.strip().lower()
        
        # Return with confidence
        confidence = 0.9 if gemma_intent == base_intent else 0.6
        
        return {
            "intent": gemma_intent if gemma_intent in ["question", "greeting", "request", "statement", "command"] else base_intent,
            "confidence": confidence,
            "base_intent": base_intent
        }
    
    async def translate_ghanaian_phrase(self, phrase: str) -> Dict[str, str]:
        """
        Translate Ghanaian phrase using Gemma.
        
        Args:
            phrase: Ghanaian English phrase
            
        Returns:
            Translation information
        """
        if not self.use_gemma or not self.gemma:
            # Use base NLP
            return {
                "original": phrase,
                "translation": phrase,
                "explanation": "Gemma not available"
            }
        
        return await self.gemma.translate_ghanaian_phrase(phrase)


class EnhancedTranslationHelper:
    """Helper for translation tasks using Gemma."""
    
    def __init__(self):
        self.gemma = get_gemma_model()
    
    async def improve_for_signing(self, text: str) -> str:
        """
        Improve text for sign language translation.
        
        Args:
            text: Original text
            
        Returns:
            Improved text
        """
        return await self.gemma.improve_translation(text)
    
    async def explain_sign_simply(self, sign_name: str, technical_description: str) -> str:
        """
        Explain a sign in simple terms.
        
        Args:
            sign_name: Name of the sign
            technical_description: Technical description
            
        Returns:
            Simple explanation
        """
        return await self.gemma.explain_sign(sign_name, technical_description)
    
    async def generate_usage_examples(self, sign_name: str, count: int = 3) -> List[str]:
        """
        Generate usage examples for a sign.
        
        Args:
            sign_name: Name of the sign
            count: Number of examples
            
        Returns:
            List of example sentences
        """
        prompt = f"""Generate {count} simple example sentences using the sign "{sign_name}" in Ghanaian Sign Language context.

Make them practical and useful for daily communication.

Examples:"""
        
        response = await self.gemma.generate_async(prompt, temperature=0.7)
        
        # Parse examples
        examples = [s.strip() for s in response.split('\n') if s.strip() and not s.strip().startswith('#')]
        return examples[:count]
    
    async def suggest_related_signs(self, sign_name: str, count: int = 5) -> List[str]:
        """
        Suggest related signs using Gemma's understanding.
        
        Args:
            sign_name: Name of the sign
            count: Number of suggestions
            
        Returns:
            List of related sign names
        """
        prompt = f"""List {count} signs that are related to or commonly used with the sign "{sign_name}" in Ghanaian Sign Language.

List only the sign names (comma-separated):"""
        
        response = await self.gemma.generate_async(prompt, temperature=0.5)
        
        # Parse suggestions
        suggestions = [s.strip().lower() for s in response.split(',')]
        return suggestions[:count]


class EnhancedLearningHelper:
    """Helper for learning content generation using Gemma."""
    
    def __init__(self):
        self.gemma = get_gemma_model()
    
    async def generate_lesson_content(
        self,
        topic: str,
        level: str = "beginner",
        signs: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Generate complete lesson content.
        
        Args:
            topic: Lesson topic
            level: Difficulty level
            signs: Optional list of signs to include
            
        Returns:
            Lesson content dictionary
        """
        signs_context = f"\nInclude these signs: {', '.join(signs)}" if signs else ""
        
        prompt = f"""Create a {level} level lesson for Ghanaian Sign Language on: {topic}{signs_context}

Include:
1. Introduction (2-3 sentences)
2. Key signs to learn (list 5-7 signs)
3. Practice exercises (3 exercises)
4. Tips for remembering

Format as structured content:"""
        
        content = await self.gemma.generate_lesson_content(topic, level)
        
        return {
            "topic": topic,
            "level": level,
            "content": content,
            "signs": signs or []
        }
    
    async def generate_quiz_questions(
        self,
        topic: str,
        count: int = 5
    ) -> List[Dict[str, any]]:
        """
        Generate quiz questions for a topic.
        
        Args:
            topic: Quiz topic
            count: Number of questions
            
        Returns:
            List of quiz questions
        """
        prompt = f"""Generate {count} multiple choice quiz questions about {topic} in Ghanaian Sign Language.

For each question provide:
- Question text
- 4 options (A, B, C, D)
- Correct answer

Format each question clearly:"""
        
        response = await self.gemma.generate_async(prompt, temperature=0.6, max_tokens=500)
        
        # Return raw response (parsing can be done by caller)
        return {
            "topic": topic,
            "questions": response,
            "count": count
        }
    
    async def generate_practice_dialogue(
        self,
        scenario: str,
        signs_to_use: List[str]
    ) -> str:
        """
        Generate a practice dialogue using specific signs.
        
        Args:
            scenario: Conversation scenario
            signs_to_use: Signs to include
            
        Returns:
            Dialogue text
        """
        signs_str = ", ".join(signs_to_use)
        
        prompt = f"""Create a simple practice dialogue for Ghanaian Sign Language learners.

Scenario: {scenario}
Use these signs: {signs_str}

Write a short conversation (4-6 exchanges) that naturally uses these signs:"""
        
        return await self.gemma.generate_async(prompt, temperature=0.7)
    
    async def provide_learning_feedback(
        self,
        user_performance: Dict[str, any]
    ) -> str:
        """
        Generate personalized learning feedback.
        
        Args:
            user_performance: Dictionary with performance metrics
            
        Returns:
            Feedback text
        """
        lessons_completed = user_performance.get("lessons_completed", 0)
        accuracy = user_performance.get("accuracy", 0)
        signs_learned = user_performance.get("signs_learned", 0)
        
        prompt = f"""Provide encouraging feedback for a Ghanaian Sign Language learner:

Progress:
- Lessons completed: {lessons_completed}
- Accuracy: {accuracy}%
- Signs learned: {signs_learned}

Write 2-3 sentences of encouraging, specific feedback:"""
        
        return await self.gemma.generate_async(prompt, temperature=0.7)


class EnhancedConversationalAssistant:
    """Conversational AI assistant for GSL learning."""
    
    def __init__(self):
        self.gemma = get_gemma_model()
        self.conversation_history = []
        self.system_message = """You are a helpful Ghanaian Sign Language (GSL) learning assistant.
You help students learn GSL by:
- Answering questions clearly and simply
- Providing encouragement
- Explaining signs and concepts
- Suggesting practice activities
- Being patient and supportive

Keep responses concise and friendly."""
    
    async def chat(self, user_message: str) -> str:
        """
        Chat with the assistant.
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response
        response = await self.gemma.chat_async(
            self.conversation_history,
            temperature=0.7
        )
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Keep history manageable (last 10 messages)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    async def answer_question(self, question: str) -> str:
        """
        Answer a specific question about GSL.
        
        Args:
            question: User's question
            
        Returns:
            Answer
        """
        prompt = f"""Answer this question about Ghanaian Sign Language clearly and simply:

Question: {question}

Answer:"""
        
        return await self.gemma.generate_async(prompt, temperature=0.5)


# Convenience functions

async def enhance_text_for_signing(text: str) -> str:
    """Quick function to improve text for signing."""
    helper = EnhancedTranslationHelper()
    return await helper.improve_for_signing(text)


async def explain_sign_simply(sign_name: str, description: str) -> str:
    """Quick function to explain a sign simply."""
    helper = EnhancedTranslationHelper()
    return await helper.explain_sign_simply(sign_name, description)


async def generate_lesson(topic: str, level: str = "beginner") -> Dict:
    """Quick function to generate a lesson."""
    helper = EnhancedLearningHelper()
    return await helper.generate_lesson_content(topic, level)


async def chat_with_assistant(message: str) -> str:
    """Quick function to chat with assistant."""
    assistant = EnhancedConversationalAssistant()
    return await assistant.chat(message)