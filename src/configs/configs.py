from enum import Enum
from typing import List, Union
from pydantic import BaseModel, Field, validator, constr


class ModelType(str, Enum):
    MISTRAL = "mistral:latest"
    GPT3 = "gpt-3.5-turbo"
    GPT3BIS = "gpt-3.5-turbo-16k"
    LLAMA2 = "llama2:13b-chat"
    LLAMA2BIS = "llama2-uncensored"


class TopologyType(str, Enum):
    SMALL_WORLD = "small-world"
    STAR = "star"
    SCALE_FREE = "scale-free"


class SimulationConfig(BaseModel):
    num_agents: int
    topic: str
    num_rounds: int
    topology: TopologyType = TopologyType.SMALL_WORLD
    model_type: ModelType = ModelType.GPT3

    temperature: float = 1.0
    small_world_k: int = 4
    small_world_p: float = 0.3


class LLMConfig(BaseModel):
    model_type: ModelType = ModelType.GPT3BIS
    temperature: float = 1.0
    presence_penalty: float = 1.0
    frequency_penalty: float = 1.0


class GraphEnvironmentConfig(BaseModel):
    topology: str
    num_agents: int
    small_world_k: int = 2
    small_world_p: float = 0.3
    scale_free_m: int = 1

    @validator("topology")
    def validate_topology(cls, value):
        valid_topologies = ["star", "small-world", "scale-free"]
        if value not in valid_topologies:
            raise ValueError(f"Invalid topology. Choose from {valid_topologies}")
        return value

    @validator("num_agents", pre=True)
    def validate_num_agents(cls, value):
        if value <= 1:
            raise ValueError("num_agents must be greater than 1")
        return value

    @validator("small_world_k", pre=True, always=True)
    def validate_small_world_k(cls, k, values):
        topology = values.get("topology")
        num_agents = values.get("num_agents")

        if topology == "small-world":
            if num_agents is not None:  # Ensure num_agents has been processed
                if num_agents <= k:
                    raise ValueError("num_agents must be greater than small_world_k")
                elif k < 2:
                    raise ValueError("small_world_k must be greater than 1")
                elif num_agents == 2 and k == 1:
                    raise ValueError("For num_agents=2, small_world_k cannot be 1")
        return k

    @validator("small_world_p")
    def validate_small_world_p(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("small_world_p must be between 0 and 1")
        return value

    @validator("scale_free_m")
    def validate_scale_free_m(cls, value):
        if value <= 0:
            raise ValueError("scale_free_m must be greater than 0")
        return value
