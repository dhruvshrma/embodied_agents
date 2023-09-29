from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.GenerativeAgent import GenerativeAgent
from langchain.schema.language_model import BaseLanguageModel
from agents.MemoryProvider import MemoryManager


class LLMInteraction:
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def chain(
        self, prompt: PromptTemplate, memory: MemoryManager, verbose: bool = False
    ) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=verbose,
            memory=memory.memory_source,
        )


class MemoryInteraction:
    def __init__(self, memory_manager: MemoryManager, llm_interaction: LLMInteraction):
        self.memory_manager = memory_manager
        self.llm_interaction = llm_interaction

    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return (
            self.llm_interaction.chain(prompt, self.memory_manager)
            .run(observation=observation)
            .strip()
        )

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.llm_interaction.chain(prompt, self.memory_manager)
            .run(entity=entity_name, observation=observation)
            .strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
            {q1}?
            Context from memory:
            {relevant_memories}
            Relevant context: 
            """
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {agent.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return (
            self.llm_interaction.chain(prompt=prompt, memory=self.memory_manager)
            .run(q1=q1, queries=[q1, q2])
            .strip()
        )

    def _generate_reaction(
        self,
        observation: str,
        suffix: str,
        agent: GenerativeAgent,
        now: Optional[datetime] = None,
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=agent.name,
            observation=observation,
            agent_status=agent.status,
        )
        consumed_tokens = self.llm_interaction.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[
            self.memory_manager.memory_source.most_recent_memories_token_key
        ] = consumed_tokens
        return (
            self.llm_interaction.chain(prompt=prompt, memory=self.memory_manager)
            .run(**kwargs)
            .strip()
        )

    def _clean_response(self, text: str, agent: GenerativeAgent) -> str:
        return re.sub(f"^{agent.name} ", "", text.strip()).strip()

    def generate_reaction(
        self, observation: str, agent: GenerativeAgent, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory_manager.memory_source.save_context(
            {},
            {
                self.memory_manager.memory_source.add_memory_key: f"{agent.name} observed "
                f"{observation} and reacted by {result}",
                self.memory_manager.memory_source.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{agent.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{agent.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(
        self, observation: str, agent: GenerativeAgent, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory_manager.memory_source.save_context(
                {},
                {
                    self.memory_manager.memory_source.add_memory_key: f"{agent.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory_manager.memory_source.now_key: now,
                },
            )
            return False, f"{agent.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory_manager.memory_source.save_context(
                {},
                {
                    self.memory_manager.memory_source.add_memory_key: f"{agent.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory_manager.memory_source.now_key: now,
                },
            )
            return True, f"{agent.name} said {response_text}"
        else:
            return False, result

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def _compute_agent_summary(self, agent: GenerativeAgent) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.llm_interaction.chain(prompt, memory=self.memory_manager)
            .run(name=agent.name, queries=[f"{agent.name}'s core characteristics"])
            .strip()
        )

    def get_summary(
        self,
        agent: GenerativeAgent,
        force_refresh: bool = False,
        now: Optional[datetime] = None,
    ) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - agent.last_refreshed).seconds
        if (
            not agent.summary
            or since_refresh >= agent.summary_refresh_seconds
            or force_refresh
        ):
            agent.summary = self._compute_agent_summary()
            agent.last_refreshed = current_time
        age = agent.age if agent.age is not None else "N/A"
        return (
            f"Name: {agent.name} (age: {age})"
            + f"\nInnate traits: {agent.traits}"
            + f"\n{agent.summary}"
        )

    def get_full_header(
        self,
        agent: GenerativeAgent,
        force_refresh: bool = False,
        now: Optional[datetime] = None,
    ) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{agent.name}'s status: {agent.status}"
