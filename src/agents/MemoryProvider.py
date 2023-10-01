from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain import PromptTemplate
from langchain.schema import BaseMemory, Document
from agents.GenerativeAgent import GenerativeAgent
from agents.GenerativeAgentMemory import GenerativeAgentMemory

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    @abstractmethod
    def get_memory(self, agent: GenerativeAgent):
        pass


class MemorySource(ABC):
    @abstractmethod
    def fetch_memories(self, observation: str) -> List[Document]:
        pass


class DictionaryMemorySource(MemorySource):
    def __init__(self, memory_data: dict):
        self.memory_data = memory_data

    def fetch_memory(self, agent_name: str):
        return self.memory_data.get(agent_name, {})


class GenerativeMemorySource(MemorySource):
    def __init__(self, gen_memory: GenerativeAgentMemory):
        self.gen_memory = gen_memory

    def fetch_memory(self, agent_name: str):
        # Logic to fetch memory for the agent from GenerativeAgentMemory
        pass


class MemoryManager(MemoryProvider):
    def __init__(self, memory_source: MemorySource):
        self.memory_source = memory_source

    def add_memories(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observations or memories to the agent's memory."""
        importance_scores = self._score_memories_importance(memory_content)

        self.aggregate_importance += max(importance_scores)
        memory_list = memory_content.split(";")
        documents = []

        for i in range(len(memory_list)):
            documents.append(
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents, current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def add_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO
