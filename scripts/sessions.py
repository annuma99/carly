"""
Session memory for Carly.

A "session" is just a running list of (question, answer) pairs, keyed by
a session_id the caller provides (one per user/conversation). This module
is intentionally storage-agnostic about WHERE it's called from -- it
doesn't know about MCP, the terminal loop, or anything else. That's the
point: session logic should be reusable regardless of which interface
sits on top of it.

Storage here is a simple in-memory dict. That means sessions are lost if
the process restarts -- fine for local testing, NOT fine for a real
deployed server (multiple users, process restarts, etc). If you deploy
this for real, swap SESSIONS for a Redis hash or a database table keyed
by session_id; nothing else in this file has to change.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SESSIONS: dict[str, list[dict]] = {}
MAX_TURNS_KEPT = 6  # how many past (question, answer) pairs to retain per session


def get_history(session_id: str) -> list[dict]:
    return SESSIONS.get(session_id, [])


def add_turn(session_id: str, question: str, answer: str) -> None:
    SESSIONS.setdefault(session_id, [])
    SESSIONS[session_id].append({"question": question, "answer": answer})
    SESSIONS[session_id] = SESSIONS[session_id][-MAX_TURNS_KEPT:]  # cap memory growth


def contextualize_query(session_id: str, question: str) -> str:
    """
    Rewrite a possibly-ambiguous follow-up question into a standalone
    question, using conversation history. This is what RETRIEVAL should
    search with -- not the raw follow-up, which often lacks enough
    context on its own (e.g. "what about the Oversight Committee?").

    If there's no history yet, this is a no-op -- skip the extra API
    call entirely, since the first question in a session is already
    standalone by definition.
    """
    history = get_history(session_id)
    if not history:
        return question

    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history
    )

    prompt = f"""Given this conversation history:

{history_text}

And this follow-up question: "{question}"

Rewrite the follow-up as a standalone question that makes sense without
the conversation history. If it's already standalone, return it unchanged.
Respond with ONLY the rewritten question, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def format_history_for_prompt(session_id: str) -> str:
    """Turn history into a block Claude can see as prior conversation context."""
    history = get_history(session_id)
    if not history:
        return ""
    turns = "\n\n".join(
        f"Student asked: {t['question']}\nYou answered: {t['answer']}" for t in history
    )
    return f"Prior conversation in this session:\n\n{turns}\n\n---\n\n"