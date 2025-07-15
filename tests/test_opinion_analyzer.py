import pytest
from agents.SimpleAgent import SimpleAgent
from personas.Persona import Persona
from configs.configs import LLMConfig, ModelType


class MockLLMClient:
    """Simple mock LLM client for testing."""
    def __init__(self, response="0.1"):
        self.response = response
    
    def generate_response(self, prompt):
        return self.response


class TestOpinionAnalyzer:
    """Test suite for OpinionAnalyzer functionality."""
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents with generic opinion traits."""
        persona1 = Persona(
            name="Alice", 
            age=30, 
            traits="change-oriented, open-minded, weakly held", 
            status="student"
        )
        persona2 = Persona(
            name="Bob", 
            age=45, 
            traits="status-quo, closed-minded, strongly held", 
            status="employed"
        )
        
        agent1 = SimpleAgent(name="Alice", agent_id=1, persona=persona1)
        agent2 = SimpleAgent(name="Bob", agent_id=2, persona=persona2)
        
        # Set initial opinions on a scale (-1 to +1)
        agent1.set_opinion(-0.5)  # Leans toward "change" side
        agent2.set_opinion(0.7)   # Leans toward "status-quo" side
        
        return [agent1, agent2]
    
    @pytest.fixture
    def sample_conversation_history(self):
        """Sample conversation history for testing."""
        return [
            (1, "Alice", "I think we should try the new approach."),
            (2, "Bob", "I'm not sure. The current method works fine."),
            (3, "Alice", "But we could improve efficiency..."),
            (4, "Bob", "Change always brings risks."),
            (5, "Alice", "The benefits outweigh the risks.")
        ]
    
    def test_opinion_analyzer_initialization(self):
        """Test OpinionAnalyzer can be initialized."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        llm_client = MockLLMClient()
        analyzer = OpinionAnalyzer(llm_client, update_frequency=3)
        
        assert analyzer.llm_client == llm_client
        assert analyzer.update_frequency == 3
    
    def test_should_update_opinions_timing(self):
        """Test opinion update timing logic."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        analyzer = OpinionAnalyzer(MockLLMClient(), update_frequency=5)
        
        # Should not update on early steps
        assert not analyzer.should_update_opinions(1)
        assert not analyzer.should_update_opinions(3)
        assert not analyzer.should_update_opinions(4)
        
        # Should update on frequency intervals
        assert analyzer.should_update_opinions(5)
        assert analyzer.should_update_opinions(10)
        assert analyzer.should_update_opinions(15)
    
    def test_analyze_opinion_changes_flexible_agent(self, sample_agents, sample_conversation_history):
        """Test opinion change analysis for flexible agent."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        # Mock LLM to return moderate opinion shift
        llm_client = MockLLMClient(response="0.2")
        analyzer = OpinionAnalyzer(llm_client)
        
        initial_opinion = sample_agents[0].get_opinion()  # Alice: -0.5
        
        analyzer.analyze_opinion_changes(sample_conversation_history, sample_agents)
        
        # Alice (weakly held opinions) should change significantly
        new_opinion = sample_agents[0].get_opinion()
        assert new_opinion != initial_opinion  # Opinion changed
        assert abs(new_opinion - initial_opinion) > 0.1  # Significant change for flexible opinions
    
    def test_analyze_opinion_changes_rigid_agent(self, sample_agents, sample_conversation_history):
        """Test opinion change analysis for rigid agent."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        # Mock LLM to return moderate opinion shift
        llm_client = MockLLMClient(response="-0.2")
        analyzer = OpinionAnalyzer(llm_client)
        
        initial_opinion = sample_agents[1].get_opinion()  # Bob: 0.7
        
        analyzer.analyze_opinion_changes(sample_conversation_history, sample_agents)
        
        # Bob (strongly held opinions) should resist change
        new_opinion = sample_agents[1].get_opinion()
        assert new_opinion != initial_opinion  # Some change occurred
        assert abs(new_opinion - initial_opinion) < 0.1  # Small change for rigid opinions
    
    def test_personality_based_resistance(self, sample_agents, sample_conversation_history):
        """Test that personality traits affect opinion change resistance."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        # Both agents get same opinion delta from LLM
        llm_client = MockLLMClient(response="0.3")
        analyzer = OpinionAnalyzer(llm_client)
        
        initial_alice = sample_agents[0].get_opinion()  # Weakly held
        initial_bob = sample_agents[1].get_opinion()    # Strongly held
        
        analyzer.analyze_opinion_changes(sample_conversation_history, sample_agents)
        
        alice_change = abs(sample_agents[0].get_opinion() - initial_alice)
        bob_change = abs(sample_agents[1].get_opinion() - initial_bob)
        
        # Alice should change more than Bob due to personality
        assert alice_change > bob_change
    
    def test_opinion_bounds_clamping(self, sample_agents, sample_conversation_history):
        """Test that opinions stay within valid bounds (-1 to 1)."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        # Mock extreme opinion change
        llm_client = MockLLMClient(response="2.0")
        analyzer = OpinionAnalyzer(llm_client)
        
        sample_agents[0].set_opinion(0.9)  # Near upper bound
        
        analyzer.analyze_opinion_changes(sample_conversation_history, sample_agents)
        
        # Opinion should be clamped to valid range
        assert -1.0 <= sample_agents[0].get_opinion() <= 1.0
    
    def test_initialize_opinion_from_persona(self, sample_agents):
        """Test opinion initialization from persona traits."""
        from opinion_dynamics.OpinionAnalyzer import OpinionAnalyzer
        
        analyzer = OpinionAnalyzer(MockLLMClient())
        
        # Alice has change-oriented traits
        alice_opinion = analyzer.initialize_opinion_from_persona(sample_agents[0])
        assert alice_opinion < 0  # Should be negative (change-oriented)
        
        # Bob has status-quo traits  
        bob_opinion = analyzer.initialize_opinion_from_persona(sample_agents[1])
        assert bob_opinion > 0  # Should be positive (status-quo)