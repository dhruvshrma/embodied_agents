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

    def get_memory(self, agent: GenerativeAgent):
        return self.memory_source.fetch_memory(agent.name)

    # Any other utility methods related to memory can be added here
