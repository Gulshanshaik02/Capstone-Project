from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)



def search_policy(query: str, k: int = 3):

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results



if __name__ == "__main__":

    query = "How can I track my order?"

    print("=" * 60)
    print("QUERY")
    print("=" * 60)

    print(query)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    results = search_policy(query, k=3)

    for index, (document, score) in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 40)

        print("Source:", document.metadata.get("source"))
        print("Score:", score)

        print("Content:")
        print(document.page_content)