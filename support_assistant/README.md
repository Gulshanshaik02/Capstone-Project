# Zepto Support Assistant

A Retrieval-Augmented Generation (RAG) based customer support assistant built using Python, LangGraph, ChromaDB, Sentence Transformers, Pydantic, and FastAPI.

## Project Overview

The system answers questions about Zepto policies using a local policy document corpus.

The application uses:

- Sentence Transformers for local embeddings
- ChromaDB for vector storage and similarity search
- LangGraph for query routing
- Pydantic for structured response validation
- FastAPI for the REST API
- A deterministic mock LLM mode as the default

No API key is required for the graded mock mode.

---

# Architecture

The complete RAG pipeline is:

```text
                   USER QUERY
                       |
                       v
                +---------------+
                | classify_intent|
                +-------+-------+
                        |
             +----------+----------+
             |                     |
             v                     v
      policy_question        general_question
             |                     |
             v                     v
   +--------------------+   +---------------+
   | retrieve_and_answer|   | direct_answer |
   +---------+----------+   +---------------+
             |
             v
         ChromaDB
             |
             v
       Top-3 chunks
             |
             v
       Top chunk snippet
             |
             +----------+
                        |
                        v
                Pydantic validation
                        |
                        v
                   JSON response


# RAG Stages

1. Ingestion

The policy documents are stored in:

data/policies/

The eight policy files are:

doc_01.txt - Delivery Policy
doc_02.txt - Returns & Refunds
doc_03.txt - Membership Tiers
doc_04.txt - Order Tracking
doc_05.txt - Order Cancellation
doc_06.txt - Damaged or Missing Items
doc_07.txt - Gift Cards
doc_08.txt - Customer Support Hours

The ingestion logic is implemented in:

src/ingest.py

The documents are loaded and divided into chunks.

2. Embedding

The project uses the local:

sentence-transformers/all-MiniLM-L6-v2

embedding model.

The embeddings are generated locally without requiring an API key.

3. Vector Storage

The generated embeddings are stored in:

chroma_db/

using ChromaDB.

The collection name is:

zepto_policies
4. Retrieval

Retrieval is implemented in:

src/retriever.py

The user's query is embedded and compared against the stored policy chunks.

The system retrieves the top 3 most similar chunks.

5. Generation

Generation and routing are implemented in:

src/graph.py

The LangGraph contains three nodes:

classify_intent
retrieve_and_answer
direct_answer

For policy questions, the retrieved context is used to create the deterministic mock response.

For general questions, the system returns a fixed response.

LangGraph Routing

The graph uses the following routing:

classify_intent
       |
       +---- policy_question ----> retrieve_and_answer
       |
       +---- general_question ---> direct_answer
Policy classification

In mock mode, the following keywords identify a policy question:

delivery
return
refund
membership
tracking
track
cancel
gift card
support hours

If one of these keywords is found in the lowercased query, the query is routed to:

retrieve_and_answer

Otherwise it is routed to:

direct_answer
Mock LLM Mode

The default mode is:

MOCK_LLM=1

or the variable can be left unset.

This mode is deterministic and does not call an external LLM.

For a policy question, the answer follows:

Based on the retrieved context: <top retrieved chunk>

For a general question:

I can only answer questions about Zepto policies right now.

The mock mode is the required graded mode.

Response Schema

Responses are validated using Pydantic.

{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}

The fields are:

Field	Description
answer	Final response
sources	Retrieved document/chunk sources
confidence	Confidence value between 0 and 1

For general questions, sources is an empty list.

Installation

Create and activate the virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Build ChromaDB

From the project root:

python src\ingest.py

Expected output:

Loaded 8 documents
Created 11 chunks
Embedding model loaded.
ChromaDB created successfully!
INGESTION COMPLETED
Test Retrieval

Run:

python src\retriever.py

Example query:

How can I track my order?

The relevant source should be:

doc_04.txt
Run LangGraph

Run:

python src\graph.py

The script demonstrates:

A policy question routed to retrieve_and_answer
A general question routed to direct_answer
Run FastAPI

Move into the source directory:

cd src

Start the application:

uvicorn api:app --reload

The API runs at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
API Endpoint
POST /ask

Request:

{
  "query": "How can I track my order?"
}

Example response:

{
  "answer": "Based on the retrieved context: Every Zepto order shows a live rider-tracking map...",
  "sources": [
    "doc_04.txt"
  ],
  "confidence": 1.0
}
Example 1 — Policy Question

Request:

{
  "query": "How can I track my order?"
}

Expected routing:

classify_intent
       |
       v
policy_question
       |
       v
retrieve_and_answer
       |
       v
ChromaDB

Expected source:

doc_04.txt
Example 2 — General Question

Request:

{
  "query": "What is Python?"
}

Expected routing:

classify_intent
       |
       v
general_question
       |
       v
direct_answer

Expected response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
Docker

Build the Docker image:

docker build -t zepto-support-assistant .

Run the container:

docker run --name zepto-support -p 7860:7860 zepto-support-assistant

The application should then be available at:

http://localhost:7860

Swagger:

http://localhost:7860/docs
Project Structure
zepto-support-assistant/
│
├── data/
│   └── policies/
│       ├── doc_01.txt
│       ├── doc_02.txt
│       ├── doc_03.txt
│       ├── doc_04.txt
│       ├── doc_05.txt
│       ├── doc_06.txt
│       ├── doc_07.txt
│       └── doc_08.txt
│
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── graph.py
│   └── api.py
│
├── tests/
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
Mock vs Real LLM

The default configuration is:

MOCK_LLM=1

The mock implementation performs deterministic keyword-based intent classification and deterministic answer generation.

An optional real LLM implementation can be enabled with:

MOCK_LLM=0

The real LLM path is optional and is not required for the graded baseline.

Conclusion

This project demonstrates a complete local RAG support assistant:

Policy Documents
       ↓
Chunking
       ↓
Sentence Transformer Embeddings
       ↓
ChromaDB
       ↓
LangGraph
       ↓
Intent Classification
       ↓
Conditional Retrieval
       ↓
Mock Answer Generation
       ↓
Pydantic Validation
       ↓
FastAPI
       ↓
Docker

The README architecture above follows the assignment's required **ingestion → embedding → retrieval → generation** explanation and identifies the relevant components. :contentReference[oaicite:2]{index=2}

## Step 8 — Save it

Press:

```text
Ctrl + S

Then we need to do the final verification, not immediately submit.

Run these tests:

python src\ingest.py
python src\retriever.py
python src\graph.py

Then start:

cd src
uvicorn api:app

And test both:

{"query":"How can I track my order?"}

and:

{"query":"What is Python?"}
One important remaining requirement

Your assignment also requires the optional real-LLM prompt template to contain the five components:

Role
Context
Task
Format
Length

plus:

Negative constraint
Few-shot example

and the real-LLM path needs retry logic for invalid JSON.                   