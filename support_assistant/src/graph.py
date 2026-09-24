import os
from pathlib import Path
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from src.retriever import search_policy
from src.schemas import AnswerResponse
from src.llm import generate_real_answer

MOCK_LLM = os.getenv("MOCK_LLM", "1")

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_docs: list
    answer: str
    sources: list[str]
    confidence: float
    response: dict


def classify_intent(state: SupportState) -> SupportState:

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "gift card",
        "support hours",
    ]

    if MOCK_LLM == "1":

        if any(keyword in query for keyword in policy_keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    
    else:

        if any(keyword in query for keyword in policy_keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    print(f"\nIntent: {intent}")

    return {
        **state,
        "intent": intent,
    }


def retrieve_and_answer(state: SupportState) -> SupportState:

    query = state["query"]

    results = search_policy(query, k=3)

    documents = [
        document
        for document, score in results
    ]

    sources = [
        document.metadata.get(
            "source",
            "unknown"
        )
        for document in documents
    ]

    if not documents:

        validated = AnswerResponse(
            answer="No relevant Zepto policy information was found.",
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "retrieved_docs": [],
            "answer": validated.answer,
            "sources": validated.sources,
            "confidence": validated.confidence,
            "response": validated.model_dump(),
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

 
    if MOCK_LLM == "1":

        top_chunk_snippet = (
            documents[0].page_content[:200].strip()
        )

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        validated = AnswerResponse(
            answer=answer,
            sources=sources,
            confidence=1.0,
        )

    else:

        try:

            validated = generate_real_answer(
                query=query,
                context=context,
                sources=sources,
            )

        except Exception as exc:

            validated = AnswerResponse(
                answer=(
                    "ERROR: The real LLM response could not "
                    "be validated after 3 attempts."
                ),
                sources=sources,
                confidence=0.0,
            )

            print(f"LLM validation error: {exc}")

    return {
        **state,
        "retrieved_docs": documents,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
        "response": validated.model_dump(),
    }

def direct_answer(state: SupportState) -> SupportState:

    answer = (
        "I can only answer questions about Zepto policies right now."
    )

    sources = []

    confidence = 1.0

    validated = AnswerResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
    )

    return {
        **state,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
        "response": validated.model_dump(),
    }


def route_intent(
    state: SupportState,
) -> Literal["retrieve_and_answer", "direct_answer"]:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():

    graph = StateGraph(SupportState)

    graph.add_node(
        "classify_intent",
        classify_intent
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer
    )

    graph.add_node(
        "direct_answer",
        direct_answer
    )

    graph.add_edge(
        START,
        "classify_intent"
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge(
        "retrieve_and_answer",
        END
    )

    graph.add_edge(
        "direct_answer",
        END
    )

    return graph.compile()


if __name__ == "__main__":

    app = build_graph()

    print("=" * 60)
    print("TEST 1 — POLICY QUESTION")
    print("=" * 60)

    query1 = "How can I track my order?"

    result1 = app.invoke({
        "query": query1
    })

    print("\nQuery:")
    print(query1)

    print("\nFinal response:")
    print(result1["response"])

    print("\n" + "=" * 60)
    print("TEST 2 — GENERAL QUESTION")
    print("=" * 60)

    query2 = "What is Python?"

    result2 = app.invoke({
        "query": query2
    })

    print("\nQuery:")
    print(query2)

    print("\nFinal response:")
    print(result2["response"])