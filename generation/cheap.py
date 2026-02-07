from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def cheap_answer_generation_phase_7_3(state: dict) -> dict:
    """
    Phase-7.3 — Cheap Answer Generation (Groq backend)
    """

    # Hard gate
    if state["generation_mode"] == "abstain":
        return {
            **state,
            "cheap_answer": None,
        }

    context_text = state["context_text"]
    query = state["query"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not fully supported by the context, "
                "say so explicitly."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            ),
        ]
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )

    response = llm.invoke(
        prompt.format_messages(
            context=context_text,
            question=query,
        )
    )

    return {
        **state,
        "cheap_answer": {
            "text": response.content.strip(), #type: ignore
            "used_context": True,
            "model": llm.model_name,
        },
    }