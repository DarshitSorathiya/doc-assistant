# ============================================================
# RAG Generation Chain & Conversational Intelligence
# ============================================================
# End-to-end RAG pipeline:
# 1. Query Contextualization (Conversational Memory)
# 2. Hybrid Retrieval + Reranking
# 3. Context & Strict Grounding Verification
# 4. LLM Generation (OpenAI, Groq, Ollama, or Local Grounded Synthesizer)
# 5. Citation Extraction & Memory Persistence
# ============================================================

import re
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.rag.retriever import document_retriever
from app.rag.citations import citation_manager, SourceCitation
from app.rag.memory import conversation_memory

logger = logging.getLogger(__name__)


class RAGResponse:
    """Standardized response from the RAG chain."""
    def __init__(
        self,
        query: str,
        answer: str,
        session_id: str,
        sources: List[SourceCitation],
        confidence: float = 1.0,
        model_used: str = "local-synthesizer",
    ):
        self.query = query
        self.answer = answer
        self.session_id = session_id
        self.sources = sources
        self.confidence = confidence
        self.model_used = model_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "session_id": self.session_id,
            "sources": [s.model_dump() for s in self.sources],
            "confidence": self.confidence,
            "model_used": self.model_used,
        }


class RAGChain:
    """Executes grounded question-answering with conversational context and citations."""

    def __init__(self):
        self.settings = get_settings()

    def contextualize_query(self, user_id: int, session_id: str, query: str) -> str:
        """Resolve pronouns and context using prior conversation history."""
        history = conversation_memory.get_history(user_id, session_id)
        if not history:
            return query

        last_user_msgs = [m.content for m in history if m.role == "user"]
        last_assistant_msgs = [m.content for m in history if m.role == "assistant"]

        if not last_user_msgs:
            return query

        # Pronoun & contextual continuation resolution
        pronouns = ["it", "this", "that", "the policy", "the document", "the account", "they", "its"]
        query_lower = query.lower()
        has_pronoun = any(p in query_lower.split() for p in pronouns) or len(query.split()) <= 4

        if has_pronoun:
            prior_topic = last_user_msgs[-1]
            return f"{query} (Context: previous topic was '{prior_topic}')"

        return query

    def generate_answer_local(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Deterministic grounded synthesizer when external LLM is offline or no API key is provided."""
        if not context_chunks:
            return "I cannot find sufficient information in your documents to answer this question."

        # Compile information from retrieved chunks
        answers = []
        q_lower = query.lower()

        stopwords = {
            "what", "is", "the", "of", "my", "in", "for", "on", "at", "to", "a",
            "an", "and", "or", "tell", "me", "about", "how", "much", "does", "do",
            "did", "can", "could", "would", "will", "have", "has", "had", "find",
            "show", "give", "get", "with", "from", "by", "as", "please", "any",
            "which", "where", "who", "when", "why",
        }
        query_words = {w for w in re.findall(r"\b\w{3,}\b", q_lower) if w not in stopwords}
        matched_sentences = []

        for chunk in context_chunks:
            text = chunk.get("text", "")
            filename = chunk.get("filename", "document")
            page = chunk.get("page_number", 1)

            sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
            for sentence in sentences:
                s_words = {w for w in re.findall(r"\b\w{3,}\b", sentence.lower()) if w not in stopwords}
                overlap = query_words.intersection(s_words)
                if overlap:
                    matched_sentences.append((len(overlap), sentence, filename, page))

        if matched_sentences:
            matched_sentences.sort(key=lambda x: x[0], reverse=True)
            # Pick top unique sentences
            seen_sentences = set()
            for count, sent, filename, page in matched_sentences[:4]:
                if sent not in seen_sentences:
                    seen_sentences.add(sent)
                    answers.append(f"{sent} [Source: {filename}, Page {page}]")

        if answers:
            return "Based on your documents:\n\n" + "\n\n".join(answers)
        
        # Fallback to top chunk snippet
        top_chunk = context_chunks[0]
        return (
            f"Based on {top_chunk.get('filename')} (Page {top_chunk.get('page_number', 1)}):\n"
            f"{top_chunk.get('text', '').strip()}"
        )

    def _call_external_llm(self, query: str, context_str: str, history_str: str) -> Optional[str]:
        """Invoke configured external LLM (OpenAI, Groq, or Ollama) with strict grounding instructions."""
        system_prompt = (
            "You are a helpful, precision-focused Personal Document Intelligence Assistant.\n"
            "Your task is to answer the user's question accurately using ONLY the provided Document Context.\n"
            "STRICT RULES:\n"
            "1. Answer strictly based on the provided context.\n"
            "2. If the context does not contain enough information to answer, state clearly: "
            "'I cannot find sufficient information in your documents to answer this question.'\n"
            "3. Do not assume or hallucinate facts not in the context.\n"
            "4. Always cite the relevant document filename and page number where you found the information."
        )

        user_content = f"Document Context:\n{context_str}\n\n"
        if history_str:
            user_content += f"Conversation History:\n{history_str}\n\n"
        user_content += f"User Question: {query}"

        # OpenAI Provider
        if self.settings.LLM_PROVIDER == "openai" and self.settings.OPENAI_API_KEY:
            try:
                import urllib.request
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.OPENAI_API_KEY}",
                }
                payload = {
                    "model": self.settings.OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                }
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"OpenAI LLM call failed: {e}. Falling back to local synthesizer.")
                return None

        # Groq Provider (OpenAI-compatible API)
        if self.settings.LLM_PROVIDER == "groq" and self.settings.GROQ_API_KEY:
            try:
                import urllib.request
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.GROQ_API_KEY}",
                }
                payload = {
                    "model": self.settings.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                }
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Groq LLM call failed: {e}. Falling back to local synthesizer.")
                return None

        # Ollama Provider
        if self.settings.LLM_PROVIDER == "ollama":
            try:
                import urllib.request
                url = f"{self.settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
                payload = {
                    "model": self.settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["message"]["content"]
            except Exception as e:
                logger.warning(f"Ollama call failed: {e}. Falling back to local synthesizer.")
                return None

        return None

    def query(
        self,
        db: Session,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """Execute complete conversational RAG query."""
        if not session_id:
            session_id = str(uuid.uuid4())

        # 1. Contextualize query with conversational memory
        effective_query = self.contextualize_query(user_id, session_id, query)

        # 2. Hybrid Retrieval + Reranking (Multi-tenant filtered by user_id)
        search_results = document_retriever.retrieve(
            db=db,
            user_id=user_id,
            query=effective_query,
            mode="hybrid",
            category=category,
            top_k=top_k,
            use_reranker=use_reranker,
        )

        chunks = [
            {
                "document_id": r.document_id,
                "filename": r.filename,
                "page_number": r.page_number,
                "text": r.text,
                "score": r.score,
                "category": r.category,
            }
            for r in search_results
        ]

        # 3. Grounding Guardrail: Check for insufficient evidence
        if citation_manager.is_insufficient_evidence(chunks, query=effective_query):
            refusal_answer = (
                "I cannot find sufficient information in your documents to answer this question."
            )
            # Record in memory
            conversation_memory.add_user_message(user_id, session_id, query)
            conversation_memory.add_assistant_message(user_id, session_id, refusal_answer, sources=[])
            return RAGResponse(
                query=query,
                answer=refusal_answer,
                session_id=session_id,
                sources=[],
                confidence=0.0,
                model_used="grounding-guardrail",
            )

        # 4. Context & History Formatting
        context_str = citation_manager.format_context_chunks(chunks)
        history_str = conversation_memory.format_history_for_prompt(user_id, session_id)

        # 5. Generate Answer via External LLM or Local Synthesizer
        answer = self._call_external_llm(effective_query, context_str, history_str)
        if self.settings.LLM_PROVIDER == "openai":
            model_name = self.settings.OPENAI_MODEL
        elif self.settings.LLM_PROVIDER == "groq":
            model_name = self.settings.GROQ_MODEL
        else:
            model_name = self.settings.OLLAMA_MODEL
        model_used = f"{self.settings.LLM_PROVIDER}:{model_name}"

        if not answer:
            answer = self.generate_answer_local(effective_query, chunks)
            model_used = "local-synthesizer"

        # 6. Extract Sources
        sources = citation_manager.extract_sources(chunks)

        # 7. Persist dialogue in Conversation Memory
        conversation_memory.add_user_message(user_id, session_id, query)
        conversation_memory.add_assistant_message(
            user_id, session_id, answer, sources=[s.model_dump() for s in sources]
        )

        return RAGResponse(
            query=query,
            answer=answer,
            session_id=session_id,
            sources=sources,
            confidence=round(float(chunks[0].get("score", 1.0)), 3) if chunks else 1.0,
            model_used=model_used,
        )


rag_chain = RAGChain()
