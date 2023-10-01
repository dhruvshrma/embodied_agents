import math
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from agents.GenerativeAgentMemory import GenerativeAgentMemory
from langchain.chat_models import ChatOpenAI


def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


def language_model_and_memory():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=1500)
    memory = GenerativeAgentMemory(
        llm=llm, memory_retriever=create_new_memory_retriever(), reflection_threshold=8
    )
    return llm, memory
