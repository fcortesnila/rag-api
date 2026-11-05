"""LLM answer generation with retrieved context."""

from openai import OpenAI

from app.models import SourceChunk

GENERATION_MODEL = "gpt-4o-mini"


def generate_answer(question: str, chunks: list[SourceChunk], client: OpenAI) -> str:
    """Generate an answer grounded in the supplied retrieved chunks."""
    context = "\n\n".join(
        f"[{chunk.source}, chunk {chunk.chunk_index}]\n{chunk.text}" for chunk in chunks
    )
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content
    return answer or "The model returned an empty answer."
