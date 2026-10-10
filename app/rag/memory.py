# ============================================================
# Conversational Memory & Session Management
# ============================================================
# Maintains session-scoped conversation history per user, allowing
# multi-turn pronoun/context resolution with strict user isolation.
# ============================================================

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in a conversation."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = Field(default_factory=time.time)
    sources: Optional[List[Dict[str, Any]]] = None


class ConversationMemory:
    """In-memory session manager with sliding window and user isolation."""

    def __init__(self, max_turns: int = 10):
        # Structure: {(user_id, session_id): [ChatMessage, ...]}
        self._sessions: Dict[tuple, List[ChatMessage]] = {}
        self.max_turns = max_turns

    def get_history(self, user_id: int, session_id: str) -> List[ChatMessage]:
        """Retrieve recent conversation history for a specific user and session."""
        key = (user_id, session_id)
        return self._sessions.get(key, [])

    def add_user_message(self, user_id: int, session_id: str, content: str):
        """Record user message in session history."""
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = []
        self._sessions[key].append(ChatMessage(role="user", content=content))
        self._trim_history(key)

    def add_assistant_message(
        self,
        user_id: int,
        session_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ):
        """Record assistant answer in session history."""
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = []
        self._sessions[key].append(
            ChatMessage(role="assistant", content=content, sources=sources)
        )
        self._trim_history(key)

    def format_history_for_prompt(self, user_id: int, session_id: str) -> str:
        """Format the dialogue history into a string for LLM contextualization."""
        history = self.get_history(user_id, session_id)
        if not history:
            return ""

        formatted_lines = []
        for msg in history[-self.max_turns:]:
            prefix = "User" if msg.role == "user" else "Assistant"
            formatted_lines.append(f"{prefix}: {msg.content}")

        return "\n".join(formatted_lines)

    def clear_session(self, user_id: int, session_id: str) -> bool:
        """Clear memory for a given user session."""
        key = (user_id, session_id)
        if key in self._sessions:
            del self._sessions[key]
            return True
        return False

    def _trim_history(self, key: tuple):
        """Ensure history does not exceed max_turns."""
        if len(self._sessions[key]) > self.max_turns * 2:
            self._sessions[key] = self._sessions[key][-self.max_turns * 2:]


conversation_memory = ConversationMemory()
