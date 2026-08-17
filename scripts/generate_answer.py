"""
Phase 4: Retrieval-augmented generation.

Pipeline: question -> hybrid_search() [Phase 3] -> format chunks into a
prompt -> Claude API -> cited, grounded answer.

This is the first point in the whole project where an LLM writes new
text. Everything before this only ever searched/ranked EXISTING text.
"""

import os
from dotenv import load_dotenv
import anthropic
from hybrid_search import hybrid_search
from sessions import contextualize_query, format_history_for_prompt, add_turn

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found -- check your .env file")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are Carly, an assistant that answers questions about the Stevens SGA governing documents.

You must answer using ONLY the excerpts provided below -- do not use any outside knowledge about student government, Stevens, or general parliamentary procedure, even if you believe it's true.

For every factual claim you make, cite the specific Article and Section it came from, in the format (Article X, Section X.X).

If the provided excerpts do not contain enough information to answer the question, say so explicitly -- do not guess, infer beyond what's stated, or fill gaps with general knowledge.

If the question asks about something the SGA governing documents likely wouldn't cover (e.g. general Institute policy, academic rules, or other organizations), say that this falls outside the SGA governing documents rather than attempting to answer."""

def format_context(chunks):
    """Turn retrieved chunk dicts into a labeled block Claude can cite from."""
    blocks = []
    for c in chunks:
        label = f"Article {c['article']}" + (f", Section {c['section']}" if c["section"] else "")
        blocks.append(f"[{label}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)

def generate_answer(question, session_id="default", top_k=15, semantic_weight=0.6, bm25_weight=0.4):
    # Step 0: turn a possibly-ambiguous follow-up into a standalone query
    # for retrieval. No-op (and no extra API call) if this is the first
    # question in the session.
    search_query = contextualize_query(session_id, question)

    # Step 1: retrieve, using the CONTEXTUALIZED query, not the raw one --
    # this is what lets "what about for the Oversight Committee?" actually
    # find the right chunks.
    retrieved = hybrid_search(search_query, top_k=top_k,
                                semantic_weight=semantic_weight,
                                bm25_weight=bm25_weight,
                                verbose=False)

    context = format_context(retrieved)
    history_block = format_history_for_prompt(session_id)

    user_message = f"""{history_block}Excerpts from the SGA governing documents:

{context}

---

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    answer = response.content[0].text
    add_turn(session_id, question, answer)  # remember this turn for next time

    print(f'\nQuestion: "{question}"')
    print(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    import uuid
    session_id = str(uuid.uuid4())  # one session for this whole terminal run

    print("Carly -- ask a question about the SGA governing documents.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        try:
            generate_answer(question, session_id=session_id)
        except Exception as e:
            print(f"Error: {e}\n")