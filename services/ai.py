import os
import logging
import asyncio
from typing import List, Dict, Optional
import openai
import httpx

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.provider = os.getenv("CHAT_MODEL_PROVIDER", "openrouter")
        self.model = os.getenv("CHAT_MODEL", "google/gemini-2.5-flash-preview-05-20:thinking")
        
        # Set up API clients
        if self.provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "openrouter":
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            self.openrouter_base_url = "https://openrouter.ai/api/v1"
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using the configured AI provider"""
        try:
            if self.provider == "openai":
                return await self._openai_generate(messages, max_tokens, temperature)
            elif self.provider == "openrouter":
                return await self._openrouter_generate(messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            raise
    
    async def _openai_generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _openrouter_generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using OpenRouter API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
                "X-Title": "Course Assistant"  # Optional, for tracking
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"OpenRouter API error {response.status_code}: {error_text}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    async def create_course_assistant_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
        course_name: str = "your course"
    ) -> List[Dict[str, str]]:
        """Create a well-structured prompt for the course assistant"""
        
        # System message
        system_message = f"""You are an intelligent course assistant for {course_name}. Your role is to help students understand course materials, answer questions, and provide educational guidance.

CAPABILITIES:
- Answer questions based on course materials (documents, slides, assignments, etc.)
- Explain complex concepts in simple terms
- Help with assignments and study strategies
- Provide relevant examples and clarifications
- Reference specific course materials when answering

GUIDELINES:
- Always base your answers on the provided course materials
- If information isn't in the materials, clearly state this limitation
- Be encouraging and supportive in your tone
- Break down complex topics into digestible parts
- Suggest follow-up questions or related topics when helpful
- If asked about grades, deadlines, or administrative matters, direct students to check with their instructor

IMPORTANT: Only use information from the provided course materials. If you don't have enough information to answer accurately, say so and suggest how the student might find the answer."""

        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history if provided
        if chat_history:
            # Include last 6 messages (3 exchanges) for context
            recent_history = chat_history[-6:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current context and query
        user_message = f"""Course Materials Context:
{context}

Student Question: {query}

Please provide a helpful, accurate response based on the course materials above."""

        messages.append({"role": "user", "content": user_message})
        
        return messages

# Global instance
ai_service = AIService() 