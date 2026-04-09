from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def process_document(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="all-minilm")
    # vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store = Chroma.from_texts(
        chunks, embeddings, persist_directory="./chroma_db"
    )
    return vector_store


def retrieve_context(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
