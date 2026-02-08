from typing import Dict, List
from langchain_groq import ChatGroq


# ============================================================
# 1. LLM-based Query Decomposition
# ============================================================

def llm_decompose_query(query: str, max_sub_queries: int = 5) -> List[str]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )

    prompt = f"""
You are a query decomposition system.

Break the user query into minimal, independent sub-questions.

Rules:
- If there is only one intent, return exactly one item
- Do NOT add new information
- Do NOT explain
- Return ONLY a numbered list
- Maximum {max_sub_queries} items

Query:
{query}
""".strip()

    response = llm.invoke(prompt).content.strip() # type: ignore

    sub_queries: List[str] = []
    for line in response.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            sub_queries.append(line.split(".", 1)[1].strip())

    return sub_queries or [query]


# ============================================================
# 2. Sample Representative Context from Vector DB
# ============================================================

def sample_vector_db_context(
    *,
    probe_query: str,
    index,
    chunks_by_embedding_id: Dict,
    model_id: str,
    k: int = 5,
) -> str:
    """
    Pulls a small, representative slice of the vector DB
    to inform query expansion.
    """
    from retrieval.phase5_2_retrieval import run_phase_5_2_retrieval

    retrieval = run_phase_5_2_retrieval(
        query=probe_query,
        index=index,
        chunks_by_embedding_id=chunks_by_embedding_id,
        model_id=model_id,
        k=k,
    )

    snippets: List[str] = []
    for c in retrieval.results:
        snippets.append(f"- {c.text[:200]}")

    return "\n".join(snippets)


# ============================================================
# 3. LLM-based DB-aware Multi Query Expansion
# ============================================================

def llm_expand_queries_with_db_context(
    *,
    sub_queries: List[str],
    index,
    chunks_by_embedding_id: Dict,
    model_id: str,
    expansions_per_query: int = 3,
) -> List[str]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    expanded_queries: List[str] = []

    for q in sub_queries:
        db_context = sample_vector_db_context(
            probe_query=q,
            index=index,
            chunks_by_embedding_id=chunks_by_embedding_id,
            model_id=model_id,
            k=5,
        )

        prompt = f"""
You are a search query generation system.

Below is a SMALL SAMPLE of the knowledge base.
Use it only to understand terminology and coverage.

Knowledge base sample:
{db_context}

Task:
Generate {expansions_per_query} diverse search queries
that would best retrieve information to answer:

Question:
{q}

Rules:
- Preserve meaning
- Use vocabulary aligned with the knowledge base
- No explanations
- Return ONLY a numbered list
""".strip()

        response = llm.invoke(prompt).content.strip() # type: ignore

        for line in response.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                expanded_queries.append(line.split(".", 1)[1].strip())

    # Deduplicate + safety trim
    seen = set()
    clean: List[str] = []
    for q in expanded_queries:
        q = " ".join(q.split()[:15])  # token safety
        if q not in seen:
            seen.add(q)
            clean.append(q)

    return clean


# ============================================================
# 4. Retrieval + Deduplication + Reranking
# ============================================================

def retrieve_and_rerank(
    *,
    expanded_queries: List[str],
    index,
    chunks_by_embedding_id: Dict,
    model_id: str,
    k: int = 5,
):
    from retrieval.phase5_2_retrieval import run_phase_5_2_retrieval

    all_chunks = []

    for q in expanded_queries:
        retrieval = run_phase_5_2_retrieval(
            query=q,
            index=index,
            chunks_by_embedding_id=chunks_by_embedding_id,
            model_id=model_id,
            k=k,
        )
        all_chunks.extend(retrieval.results)

    # Deduplicate by chunk_id (keep highest score)
    best_by_id = {}
    for c in all_chunks:
        if c.chunk_id not in best_by_id or c.score > best_by_id[c.chunk_id].score:
            best_by_id[c.chunk_id] = c

    deduped = list(best_by_id.values())

    # Rerank by similarity score
    reranked = sorted(deduped, key=lambda c: c.score, reverse=True)

    return reranked


# ============================================================
# 5. Context Assembly (Token-aware)
# ============================================================

def assemble_context(chunks, max_tokens: int = 500) -> str:
    used_tokens = 0
    context_parts: List[str] = []

    for c in chunks:
        token_len = len(c.text.split())
        if used_tokens + token_len > max_tokens:
            break
        context_parts.append(c.text)
        used_tokens += token_len

    return "\n\n".join(context_parts)


# ============================================================
# 6. Final Answer Generation
# ============================================================

def generate_answer(context: str, query: str) -> str:
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
    return response.content.strip()  # type: ignore


# ============================================================
# 7. Orchestrator — Single-pass RAG
# ============================================================

def run_simple_rag(state: Dict) -> Dict:
    query: str = state["query"]

    # 1. Decompose query
    sub_queries = llm_decompose_query(query)

    # 2. Expand queries with DB awareness
    expanded_queries = llm_expand_queries_with_db_context(
        sub_queries=sub_queries,
        index=state["index"],
        chunks_by_embedding_id=state["chunks_by_embedding_id"],
        model_id=state["model_id"],
    )

    # 3. Retrieve + rerank
    reranked_chunks = retrieve_and_rerank(
        expanded_queries=expanded_queries,
        index=state["index"],
        chunks_by_embedding_id=state["chunks_by_embedding_id"],
        model_id=state["model_id"],
    )

    # 4. Assemble context
    context_text = assemble_context(reranked_chunks)

    # 5. Generate answer
    answer = generate_answer(context_text, query)

    return {
        "query": query,
        "sub_queries": sub_queries,
        "expanded_queries": expanded_queries,
        "final_chunks": reranked_chunks,
        "context_text": context_text,
        "answer": answer,
    }