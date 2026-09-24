from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = BASE_DIR / "data" / "policies"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"



def load_documents():
    documents = []

    for file_path in sorted(POLICY_DIR.glob("doc_*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            print(f"WARNING: {file_path.name} is empty")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    print(f"Loaded {len(documents)} documents")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks



def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings



def create_vector_store(chunks, embeddings):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store



def main():

    print("=" * 50)
    print("ZEpto Support Assistant - Document Ingestion")
    print("=" * 50)

    
    documents = load_documents()

    if not documents:
        print("No policy documents found.")
        return

    
    chunks = split_documents(documents)

    
    print("Creating embedding model...")
    embeddings = create_embeddings()

    print("Embedding model loaded.")

    
    print("Creating ChromaDB vector store...")

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    print("ChromaDB created successfully!")

    print("=" * 50)
    print("INGESTION COMPLETED")
    print("=" * 50)

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Database  : {CHROMA_DIR}")


if __name__ == "__main__":
    main()