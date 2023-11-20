import math
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from agents.GenerativeAgentMemory import GenerativeAgentMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama


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


def ChatOpenAI_init_func(model_name: str):
    return ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=1500)


def Ollama_init_func(model_name: str):
    llm = Ollama(model=model_name, temperature=0.0)
    return llm


MODEL_MAP = {
    "gpt-3.5-turbo": (ChatOpenAI, ChatOpenAI_init_func),
    "gpt-3.5-turbo-16k-0613": (ChatOpenAI, ChatOpenAI_init_func),
    "llama2:13b": (
        Ollama,
        Ollama_init_func,
    ),
    "mistral": (
        Ollama,
        Ollama_init_func,
    ),
    "llama2-uncensored": (
        Ollama,
        Ollama_init_func,
    ),
}
