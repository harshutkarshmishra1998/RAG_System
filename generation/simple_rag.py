from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from langchain_groq import ChatGroq


# ============================================================
# 1. QUERY DECOMPOSITION (LLM)
# ============================================================

def llm_decompose_query(query: str, max_sub_queries: int = 5) -> List[str]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )

    prompt = f"""
Break the user query into minimal independent sub-questions.

Rules:
- If single intent → return 1 item
- Do NOT add new information
- Do NOT explain
- Return ONLY a numbered list
- Max {max_sub_queries} items

Query:
{query}
""".strip()

    response = llm.invoke(prompt).content.strip() #type: ignore

    sub_queries: List[str] = []
    for line in response.splitlines():
        if line[:1].isdigit():
            sub_queries.append(line.split(".", 1)[1].strip())

    return sub_queries or [query]


# ============================================================
# 2. MULTI-QUERY EXPANSION (LLM)
# ============================================================

def llm_expand_queries(
    sub_queries: List[str],
    expansions_per_query: int = 3,
) -> List[str]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    expanded: List[str] = []

    for q in sub_queries:
        prompt = f"""
Generate {expansions_per_query} diverse search queries
to retrieve information that answers:

{q}

Rules:
- Preserve meaning
- No explanations
- Return ONLY a numbered list
""".strip()

        response = llm.invoke(prompt).content.strip() #type: ignore

        for line in response.splitlines():
            if line[:1].isdigit():
                expanded.append(line.split(".", 1)[1].strip())

    # Deduplicate
    seen = set()
    deduped: List[str] = []
    for q in expanded:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped


# ============================================================
# 3. QUERY FUSION STRUCTURE (IMMUTABLE)
# ============================================================

@dataclass(frozen=True)
class FusedChunk:
    chunk: object        # RetrievedChunk
    fused_score: float
    frequency: int


# ============================================================
# 4. MULTI-QUERY RETRIEVAL + QUERY FUSION
# ============================================================

def retrieve_with_query_fusion(
    *,
    queries: List[str],
    index,
    chunks_by_embedding_id: Dict,
    model_id: str,
    k: int = 5,
) -> List[FusedChunk]:
    """
    Correct multi-query retrieval with score fusion.
    Does NOT mutate frozen RetrievedChunk.
    """

    from retrieval.phase5_2_retrieval import run_phase_5_2_retrieval

    score_buckets = defaultdict(list)   # chunk_id -> scores
    chunk_lookup = {}

    for q in queries:
        retrieval = run_phase_5_2_retrieval(
            query=q,
            index=index,
            chunks_by_embedding_id=chunks_by_embedding_id,
            model_id=model_id,
            k=k,
        )

        for c in retrieval.results:
            score_buckets[c.chunk_id].append(c.score)
            chunk_lookup[c.chunk_id] = c

    fused_chunks: List[FusedChunk] = []

    for chunk_id, scores in score_buckets.items():
        frequency = len(scores)
        fused_score = sum(scores) * frequency

        fused_chunks.append(
            FusedChunk(
                chunk=chunk_lookup[chunk_id],
                fused_score=fused_score,
                frequency=frequency,
            )
        )

    fused_chunks.sort(key=lambda x: x.fused_score, reverse=True)
    return fused_chunks


# ============================================================
# 5. CONTEXT ASSEMBLY
# ============================================================

def assemble_context(
    fused_chunks: List[FusedChunk],
    max_tokens: int = 500,
) -> str:
    used_tokens = 0
    parts: List[str] = []

    for fc in fused_chunks:
        text = fc.chunk.text #type: ignore
        tokens = len(text.split())

        if used_tokens + tokens > max_tokens:
            break

        parts.append(text)
        used_tokens += tokens

    return "\n\n".join(parts)


# ============================================================
# 6. FINAL ANSWER GENERATION
# ============================================================

def generate_answer(context: str, query: str) -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    prompt = f"""
Answer the question using ONLY the context below.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{query}
""".strip()

    return llm.invoke(prompt).content.strip() #type: ignore


# ============================================================
# 7. ORCHESTRATOR (SINGLE-PASS RAG)
# ============================================================

def run_simple_rag(state: Dict) -> Dict:
    query = state["query"]

    # ---- 1. Decompose ----
    sub_queries = llm_decompose_query(query)

    # ---- 2. Expand ----
    expanded_queries = llm_expand_queries(sub_queries)

    print("\n================ QUERY ANALYSIS =================")
    print("Original query:", query)

    print("\nSub-queries count:", len(sub_queries))
    for i, q in enumerate(sub_queries, 1):
        print(f" {i}. {q}")

    print("\nExpanded queries count:", len(expanded_queries))
    for i, q in enumerate(expanded_queries, 1):
        print(f" {i}. {q}")

    # ---- 3. Retrieve with QUERY FUSION ----
    fused_chunks = retrieve_with_query_fusion(
        queries=expanded_queries,
        index=state["index"],
        chunks_by_embedding_id=state["chunks_by_embedding_id"],
        model_id=state["model_id"],
    )

    print("\n================ QUERY FUSION DEBUG =================")
    for i, fc in enumerate(fused_chunks[:10], 1):
        print(
            f"{i}. chunk_id={fc.chunk.chunk_id} " #type: ignore
            f"freq={fc.frequency} "
            f"fused_score={fc.fused_score:.4f}"
        )

    # ---- 4. Assemble context ----
    context = assemble_context(fused_chunks)

    # ---- 5. Generate answer ----
    answer = generate_answer(context, query)

    return {
        "query": query,
        "sub_queries": sub_queries,
        "expanded_queries": expanded_queries,
        "fused_chunks": fused_chunks,
        "context": context,
        "answer": answer,
    }