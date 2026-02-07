# generation/deep/answer.py

from typing import Dict
from langchain_groq import ChatGroq

def deep_answer_generation_phase_7_4_6(state: Dict) -> Dict:
    """
    Phase-7.4.6 — Deep Answer Generation
    Uses deep_context_text produced in Phase-7.4.5
    """

    context = state.get("deep_context_text")
    query = state["query"]

    if not context:
        return {
            **state,
            "deep_answer": None,
        }

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    prompt = f"""
You are an expert assistant.

Use ONLY the context below to answer the question.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{query}
""".strip()

    response = llm.invoke(prompt)

    return {
        **state,
        "deep_answer": {
            "text": response.content,
            "used_context": True,
            "model": "llama-3.1-8b-instant",
        },
    }
