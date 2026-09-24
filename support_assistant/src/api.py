from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.graph import build_graph

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0",
)



class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )



class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )



graph = build_graph()


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    result = graph.invoke({
        "query": request.query
    })

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )



@app.get("/")
def root():

    return {
        "message": "Zepto Support Assistant is running"
    }