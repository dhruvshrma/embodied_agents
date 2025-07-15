import random
from typing import List, Tuple
from agents.SimpleAgent import SimpleAgent


class OpinionAnalyzer:
    """Analyzes conversation history and updates agent opinions based on dialogue."""

    def __init__(self, llm_client, update_frequency: int = 5):
        """
        Initialize the OpinionAnalyzer.

        Args:
            llm_client: Client for LLM API calls
            update_frequency: How often to update opinions (every N steps)
        """
        self.llm_client = llm_client
        self.update_frequency = update_frequency

    def should_update_opinions(self, step_count: int) -> bool:
        """
        Determine if opinions should be updated based on step count.

        Args:
            step_count: Current simulation step

        Returns:
            True if opinions should be updated
        """
        return step_count % self.update_frequency == 0

    def analyze_opinion_changes(
        self, conversation_history: List[Tuple], agents: List[SimpleAgent]
    ) -> None:
        """
        Analyze conversation history and update agent opinions.

        Args:
            conversation_history: List of (step, speaker_name, message) tuples
            agents: List of agents to potentially update
        """
        for agent in agents:
            # Get opinion delta from LLM analysis
            opinion_delta = self._get_opinion_delta_from_llm(
                agent, conversation_history
            )

            # Apply personality-based resistance
            adjusted_delta = self._apply_personality_resistance(agent, opinion_delta)

            # Update agent opinion
            new_opinion = agent.get_opinion() + adjusted_delta

            # Clamp to valid range (-1 to 1)
            new_opinion = max(-1.0, min(1.0, new_opinion))

            agent.set_opinion(new_opinion)

    def initialize_opinion_from_persona(self, agent: SimpleAgent) -> float:
        """
        Initialize agent opinion based on their persona traits.

        Args:
            agent: Agent to initialize opinion for

        Returns:
            Initial opinion value (-1 to 1)
        """
        base_opinion = 0.0

        if agent.persona and agent.persona.traits:
            traits = agent.persona.traits.lower()

            # Map traits to opinion tendencies
            if "change-oriented" in traits:
                base_opinion = -0.7
            elif "status-quo" in traits:
                base_opinion = 0.7

            # Add some random variation
            base_opinion += random.uniform(-0.2, 0.2)

            # Clamp to valid range
            base_opinion = max(-1.0, min(1.0, base_opinion))

        return base_opinion

    def _get_opinion_delta_from_llm(
        self, agent: SimpleAgent, conversation_history: List[Tuple]
    ) -> float:
        """
        Get opinion change from LLM analysis.

        Args:
            agent: Agent to analyze
            conversation_history: Recent conversation history

        Returns:
            Opinion change delta (-1 to 1)
        """
        # Create prompt for LLM
        recent_messages = self._format_conversation_for_prompt(conversation_history)

        prompt = f"""
        Analyze how {agent.name}'s opinion might change based on this conversation.
        Agent traits: {agent.persona.traits if agent.persona else 'unknown'}
        
        Recent conversation:
        {recent_messages}
        
        Rate the opinion change from -1 (strongly moved toward change) to +1 (strongly moved toward status-quo).
        Return only a number between -1 and 1.
        """

        response = self.llm_client.generate_response(prompt)

        try:
            # Parse response as float
            delta = float(response.strip())
            # Clamp to valid range
            return max(-1.0, min(1.0, delta))
        except ValueError:
            # If parsing fails, return no change
            return 0.0

    def _apply_personality_resistance(
        self, agent: SimpleAgent, opinion_delta: float
    ) -> float:
        """
        Apply personality-based resistance to opinion change.

        Args:
            agent: Agent whose personality affects resistance
            opinion_delta: Raw opinion change from LLM

        Returns:
            Adjusted opinion change
        """
        if not agent.persona or not agent.persona.traits:
            return opinion_delta

        traits = agent.persona.traits.lower()

        # Apply resistance based on personality traits
        if "strongly held" in traits:
            # Strongly held opinions resist change
            return opinion_delta * 0.3
        elif "weakly held" in traits:
            # Weakly held opinions change more easily
            return opinion_delta * 1.5
        elif "closed-minded" in traits:
            # Closed-minded agents resist change
            return opinion_delta * 0.4
        elif "open-minded" in traits:
            # Open-minded agents change more easily
            return opinion_delta * 1.2

        # Default: no modification
        return opinion_delta

    def _format_conversation_for_prompt(self, conversation_history: List[Tuple]) -> str:
        """
        Format conversation history for LLM prompt.

        Args:
            conversation_history: List of (step, speaker_name, message) tuples

        Returns:
            Formatted conversation string
        """
        formatted_messages = []
        for step, speaker_name, message in conversation_history:
            formatted_messages.append(f"{speaker_name}: {message}")

        return "\n".join(formatted_messages)
