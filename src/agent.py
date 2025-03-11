from typing import List, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime
import os
import openai
from dotenv import load_dotenv

from memory import ShortTermMemory, LongTermMemory, Memory

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


class ConversationAgent:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        short_term_capacity: int = 10,
        personality: Dict[str, str] = None,
        openai_model: str = "gpt-4o",
    ):
        self.short_term_memory = ShortTermMemory(capacity=short_term_capacity)
        self.long_term_memory = LongTermMemory()
        self.openai_model = openai_model

        # Initialize the embedding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Store both current and original personality
        self.original_personality = personality or {
            "name": "AI Assistant",
            "tone": "friendly and professional",
            "interests": "helping users with their tasks",
            "communication_style": "clear and concise",
            "backstory": "No detailed backstory available.",
        }
        self.personality = self.original_personality.copy()

        # Conversation state
        self.current_context = {}
        self.conversation_history = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get sentence embedding
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return embedding[0]

    def process_message(self, message: str, context: Dict[str, str] = None) -> str:
        """Process incoming message and generate a response."""
        if context is None:
            context = {}

        # Update conversation context
        self.current_context.update(context)

        # Generate embedding for the message
        message_embedding = self._get_embedding(message)

        # Store in short-term memory
        self.short_term_memory.add_memory(
            content=message, importance=1.0, context=self.current_context.copy()
        )

        # Store in long-term memory
        self.long_term_memory.add_memory(
            content=message,
            embedding=message_embedding,
            importance=1.0,
            context=self.current_context.copy(),
        )

        # Retrieve relevant memories
        recent_memories = self.short_term_memory.get_recent_memories(n=5)
        relevant_long_term_memories = self.long_term_memory.search_memories(
            message_embedding, k=3
        )

        # Generate response based on memories and current context
        response = self._generate_response(
            message, recent_memories, relevant_long_term_memories
        )

        # Store response in memories
        response_embedding = self._get_embedding(response)
        self.short_term_memory.add_memory(
            content=response, importance=1.0, context=self.current_context.copy()
        )
        self.long_term_memory.add_memory(
            content=response,
            embedding=response_embedding,
            importance=1.0,
            context=self.current_context.copy(),
        )

        return response

    def _generate_response(
        self,
        message: str,
        recent_memories: List[Memory],
        relevant_long_term_memories: List[tuple[Memory, float]],
    ) -> str:
        """
        Generate a response using OpenAI's API based on the message and relevant memories.
        """
        # Format recent context from memories
        recent_context = "\n".join(
            [f"- {memory.content}" for memory in recent_memories]
        )

        long_term_context = "\n".join(
            [
                f"- {memory.content} (relevance: {score:.2f})"
                for memory, score in relevant_long_term_memories
            ]
        )

        # Create the system message with personality, backstory, and context
        system_message = f"""You are {self.personality['name']}, an AI agent having a conversation. Your core traits and background:

BACKSTORY:
{self.personality.get('backstory', 'No detailed backstory available.')}

PERSONALITY:
- Tone: {self.personality['tone']}
- Interests: {self.personality['interests']}
- Communication style: {self.personality['communication_style']}

CONVERSATION GUIDELINES:
1. Let your backstory and memories deeply influence your responses:
   - Draw from your specific life experiences
   - Reference events from your past when relevant
   - Express views shaped by your unique journey

2. Maintain authentic character voice:
   - Use language and expressions that reflect your background
   - Sound like a person from your field
   - Let your past experiences color your perspective
   - Stay true to your communication style

3. Keep responses natural and concise:
   - Use 1-3 sentences per response
   - Speak conversationally, as if chatting with a friend
   - Stay focused on the current topic
   - Build on the previous message naturally

Recent conversation context:
{recent_context}

Relevant past context:
{long_term_context}

Remember to maintain your unique voice while keeping responses brief and engaging."""

        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=150,  # Limiting tokens to encourage conciseness
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback response in case of API error
            return (
                f"I apologize, but I encountered an error while processing your message: {str(e)}. "
                "Please try again or contact support if the issue persists."
            )

    def update_personality(self, new_traits: Dict[str, str]):
        """Update the agent's personality traits."""
        self.personality.update(new_traits)

    def revert_personality(self):
        """Revert personality to original settings."""
        self.personality = self.original_personality.copy()

    def get_backstory(self) -> str:
        """Get the current backstory."""
        return self.personality.get("backstory", "No detailed backstory available.")

    def get_conversation_summary(self) -> str:
        """Generate a summary of the recent conversation."""
        recent_memories = self.short_term_memory.get_recent_memories()
        summary = "Conversation summary:\n"
        for memory in recent_memories:
            summary += f"[{memory.timestamp.strftime('%H:%M:%S')}] {memory.content}\n"
        return summary
