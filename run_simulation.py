#!/usr/bin/env python3
"""
Interactive command-line simulation runner for embodied agents.

This script provides a command-line interface to run agent simulations with 
real-time colored output showing agent exchanges and conversations.
"""

import argparse
import sys
import os
from typing import Dict, List
import signal

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.configs import SimulationConfig, ModelType, TopologyType
from simulation.SimulationRunner import SimulationRunner
from interactions.DialogueSimulation import DialogueSimulator
from utils.event_handler import EventHandler, AgentSpoke

class ColoredOutput:
    """ANSI color codes for terminal output."""
    
    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Extended colors for more agents
    COLORS = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Magenta
        '\033[96m',  # Cyan
        '\033[97m',  # White
        '\033[31m',  # Dark Red
        '\033[32m',  # Dark Green
        '\033[33m',  # Dark Yellow
        '\033[34m',  # Dark Blue
        '\033[35m',  # Dark Magenta
        '\033[36m',  # Dark Cyan
        '\033[37m',  # Gray
        '\033[90m',  # Dark Gray
    ]
    
    @classmethod
    def get_color(cls, index: int) -> str:
        """Get a color for agent at given index."""
        return cls.COLORS[index % len(cls.COLORS)]

class InteractiveSimulationRunner:
    """Enhanced simulation runner with colored command-line output."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.runner = SimulationRunner(config=config, interaction_model=DialogueSimulator)
        self.agent_colors: Dict[str, str] = {}
        self.round_count = 0
        self.setup_signal_handlers()
        self.setup_agent_colors()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print(f"\n{ColoredOutput.YELLOW}Simulation interrupted by user.{ColoredOutput.RESET}")
        print(f"{ColoredOutput.CYAN}Final simulation stats:{ColoredOutput.RESET}")
        self.print_simulation_summary()
        sys.exit(0)
    
    def setup_agent_colors(self):
        """Assign colors to each agent."""
        agents = self.runner.interaction_model.agents
        for i, agent in enumerate(agents):
            self.agent_colors[agent.name] = ColoredOutput.get_color(i)
    
    def print_header(self):
        """Print simulation header with configuration."""
        print(f"{ColoredOutput.BOLD}{ColoredOutput.CYAN}="*60)
        print(f"🤖 EMBODIED AGENTS SIMULATION")
        print(f"="*60 + ColoredOutput.RESET)
        print(f"{ColoredOutput.BOLD}Configuration:{ColoredOutput.RESET}")
        print(f"  📊 Topic: {ColoredOutput.YELLOW}{self.config.topic}{ColoredOutput.RESET}")
        print(f"  👥 Agents: {ColoredOutput.GREEN}{self.config.num_agents}{ColoredOutput.RESET}")
        print(f"  🔄 Rounds: {ColoredOutput.BLUE}{self.config.num_rounds}{ColoredOutput.RESET}")
        print(f"  🌐 Topology: {ColoredOutput.MAGENTA}{self.config.topology}{ColoredOutput.RESET}")
        print(f"  🤖 Model: {ColoredOutput.CYAN}{self.config.model_type.value}{ColoredOutput.RESET}")
        print(f"  🌡️  Temperature: {ColoredOutput.RED}{self.config.temperature}{ColoredOutput.RESET}")
        print()
        
        # Print agent roster with colors and personas
        print(f"{ColoredOutput.BOLD}Agent Roster:{ColoredOutput.RESET}")
        for agent in self.runner.interaction_model.agents:
            color = self.agent_colors[agent.name]
            persona_info = ""
            if hasattr(agent, 'persona') and agent.persona:
                persona_info = f" ({agent.persona.age}yo, {agent.persona.status})"
            print(f"  {color}● {agent.name}{persona_info}{ColoredOutput.RESET}")
            if hasattr(agent, 'persona') and agent.persona and agent.persona.traits:
                print(f"    {color}  → {agent.persona.traits}{ColoredOutput.RESET}")
        print(f"{ColoredOutput.CYAN}{'='*60}{ColoredOutput.RESET}")
        print()
    
    def print_round_header(self, round_num: int):
        """Print header for each round."""
        print(f"\n{ColoredOutput.BOLD}{ColoredOutput.BLUE}🔄 ROUND {round_num}{ColoredOutput.RESET}")
        print(f"{ColoredOutput.BLUE}{'─'*40}{ColoredOutput.RESET}")
    
    def print_agent_message(self, agent_name: str, message: str):
        """Print agent message with colored output."""
        color = self.agent_colors.get(agent_name, ColoredOutput.WHITE)
        
        # Format the message nicely
        print(f"{color}{ColoredOutput.BOLD}{agent_name}:{ColoredOutput.RESET}")
        
        # Indent the message content
        for line in message.split('\n'):
            if line.strip():
                print(f"{color}  {line}{ColoredOutput.RESET}")
        print()
    
    def run_simulation(self):
        """Run the simulation with interactive output."""
        self.print_header()
        
        try:
            # Start simulation
            print(f"{ColoredOutput.GREEN}🚀 Starting simulation...{ColoredOutput.RESET}\n")
            
            # Initialize with mediator if needed
            if hasattr(self.runner.interaction_model, 'inject'):
                self.runner.interaction_model.inject(None, "")
            
            # Run simulation rounds
            for round_num in range(1, self.config.num_rounds + 1):
                self.round_count = round_num
                self.print_round_header(round_num)
                
                # Execute one round
                result = self.runner.interaction_model.step()
                
                # Display the result
                if result and hasattr(result, 'agent_name') and hasattr(result, 'message'):
                    self.print_agent_message(result.agent_name, result.message)
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Handle tuple format (agent_name, message)
                    agent_name, message = result[0], result[1]
                    self.print_agent_message(agent_name, message)
                
                # Small delay to make it easier to follow
                import time
                time.sleep(0.5)
            
            print(f"\n{ColoredOutput.GREEN}✅ Simulation completed successfully!{ColoredOutput.RESET}")
            self.print_simulation_summary()
            
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        except Exception as e:
            print(f"\n{ColoredOutput.RED}❌ Simulation failed: {str(e)}{ColoredOutput.RESET}")
            raise
    
    def print_simulation_summary(self):
        """Print summary statistics."""
        print(f"\n{ColoredOutput.BOLD}{ColoredOutput.CYAN}📊 SIMULATION SUMMARY{ColoredOutput.RESET}")
        print(f"{ColoredOutput.CYAN}{'='*60}{ColoredOutput.RESET}")
        print(f"  Completed rounds: {ColoredOutput.GREEN}{self.round_count}{ColoredOutput.RESET}")
        print(f"  Total agents: {ColoredOutput.BLUE}{len(self.runner.interaction_model.agents)}{ColoredOutput.RESET}")
        
        # Show agent message counts if available
        if hasattr(self.runner.interaction_model, 'history'):
            history = self.runner.interaction_model.history
            print(f"  Total messages: {ColoredOutput.YELLOW}{len(history)}{ColoredOutput.RESET}")
        
        print(f"  Topic: {ColoredOutput.MAGENTA}{self.config.topic}{ColoredOutput.RESET}")
        
        # Show opinion dynamics if available
        self.print_opinion_summary()
    
    def print_opinion_summary(self):
        """Print agent opinions summary table."""
        print(f"\n{ColoredOutput.BOLD}{ColoredOutput.CYAN}🧠 AGENT OPINIONS SUMMARY{ColoredOutput.RESET}")
        print(f"{ColoredOutput.CYAN}{'='*60}{ColoredOutput.RESET}")
        
        agents = self.runner.interaction_model.agents
        
        # Print table header
        print(f"{ColoredOutput.BOLD}{'Agent Name':<15} {'Opinion':<10} {'Tendency':<15} {'Traits':<25}{ColoredOutput.RESET}")
        print(f"{ColoredOutput.CYAN}{'-'*65}{ColoredOutput.RESET}")
        
        for agent in agents:
            color = self.agent_colors.get(agent.name, ColoredOutput.WHITE)
            opinion = agent.get_opinion()
            
            # Determine opinion tendency
            if opinion < -0.3:
                tendency = "Change-Oriented"
                tendency_color = ColoredOutput.GREEN
            elif opinion > 0.3:
                tendency = "Status-Quo"
                tendency_color = ColoredOutput.RED
            else:
                tendency = "Neutral"
                tendency_color = ColoredOutput.YELLOW
            
            # Format opinion value
            opinion_str = f"{opinion:+.2f}"
            
            # Truncate traits for display
            traits = ""
            if hasattr(agent, 'persona') and agent.persona and agent.persona.traits:
                traits = agent.persona.traits[:22] + "..." if len(agent.persona.traits) > 25 else agent.persona.traits
            
            print(f"{color}{agent.name:<15}{ColoredOutput.RESET} "
                  f"{opinion_str:<10} "
                  f"{tendency_color}{tendency:<15}{ColoredOutput.RESET} "
                  f"{traits:<25}")
        
        # Print opinion statistics
        opinions = [agent.get_opinion() for agent in agents]
        avg_opinion = sum(opinions) / len(opinions)
        opinion_range = max(opinions) - min(opinions)
        
        print(f"\n{ColoredOutput.BOLD}Opinion Statistics:{ColoredOutput.RESET}")
        print(f"  Average Opinion: {ColoredOutput.CYAN}{avg_opinion:+.3f}{ColoredOutput.RESET}")
        print(f"  Opinion Range: {ColoredOutput.YELLOW}{opinion_range:.3f}{ColoredOutput.RESET}")
        
        # Show polarization level
        if opinion_range > 1.0:
            polarization = "High"
            polar_color = ColoredOutput.RED
        elif opinion_range > 0.5:
            polarization = "Medium"
            polar_color = ColoredOutput.YELLOW
        else:
            polarization = "Low"
            polar_color = ColoredOutput.GREEN
        
        print(f"  Polarization: {polar_color}{polarization}{ColoredOutput.RESET}")
        
        # Show topology-specific insights
        topology = self.config.topology if hasattr(self.config, 'topology') else "unknown"
        print(f"\n{ColoredOutput.BOLD}Network Effect ({topology}):{ColoredOutput.RESET}")
        
        if hasattr(self.runner.interaction_model, 'opinion_analyzer'):
            if self.runner.interaction_model.opinion_analyzer:
                update_freq = self.runner.interaction_model.opinion_analyzer.update_frequency
                print(f"  Opinion updates occurred every {ColoredOutput.BLUE}{update_freq}{ColoredOutput.RESET} rounds")
            else:
                print(f"  {ColoredOutput.YELLOW}No opinion tracking configured{ColoredOutput.RESET}")
        else:
            print(f"  {ColoredOutput.YELLOW}Opinion analyzer not available{ColoredOutput.RESET}")

def create_config_from_args(args) -> SimulationConfig:
    """Create simulation configuration from command line arguments."""
    
    # Parse model type
    try:
        model_type = ModelType(args.model)
    except ValueError:
        print(f"Invalid model type: {args.model}")
        print(f"Available models: {[m.value for m in ModelType]}")
        sys.exit(1)
    
    # Parse topology
    try:
        topology = TopologyType(args.topology)
    except ValueError:
        print(f"Invalid topology: {args.topology}")
        print(f"Available topologies: {[t.value for t in TopologyType]}")
        sys.exit(1)
    
    return SimulationConfig(
        num_agents=args.agents,
        topic=args.topic,
        num_rounds=args.rounds,
        model_type=model_type,
        topology=topology,
        temperature=args.temperature,
        opinion_update_frequency=args.opinion_frequency
    )

def main():
    """Main entry point for the simulation runner."""
    parser = argparse.ArgumentParser(
        description="Run interactive embodied agents simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --topic "climate change" --agents 5 --rounds 10
  %(prog)s --topic "AI ethics" --model gpt-3.5-turbo --topology star
  %(prog)s --topic "philosophy" --agents 3 --rounds 5 --temperature 0.7
  %(prog)s --topic "technology adoption" --agents 4 --rounds 15 --opinion-frequency 3
        """
    )
    
    # Required arguments
    parser.add_argument('--topic', '-t', required=True, 
                       help='Discussion topic for the simulation')
    
    # Optional arguments with defaults
    parser.add_argument('--agents', '-a', type=int, default=3,
                       help='Number of agents (default: 3)')
    parser.add_argument('--rounds', '-r', type=int, default=5,
                       help='Number of simulation rounds (default: 5)')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo',
                       help='LLM model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--topology', default='star',
                       help='Network topology (default: star)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='LLM temperature (default: 0.7)')
    parser.add_argument('--opinion-frequency', type=int, default=5,
                       help='Update agent opinions every N rounds (default: 5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.agents < 2:
        print("Error: Need at least 2 agents for simulation")
        sys.exit(1)
    
    if args.rounds < 1:
        print("Error: Need at least 1 round for simulation")
        sys.exit(1)
    
    if not (0.0 <= args.temperature <= 2.0):
        print("Error: Temperature must be between 0.0 and 2.0")
        sys.exit(1)
    
    # Create configuration
    try:
        config = create_config_from_args(args)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Check for required environment variables
    if config.model_type in [ModelType.GPT3, ModelType.GPT3BIS]:
        if not os.environ.get('OPENAI_API_KEY'):
            print(f"{ColoredOutput.RED}Error: OPENAI_API_KEY environment variable is required for OpenAI models{ColoredOutput.RESET}")
            print(f"{ColoredOutput.YELLOW}Please set it with: export OPENAI_API_KEY='your-api-key'{ColoredOutput.RESET}")
            sys.exit(1)
    
    # Run simulation
    try:
        runner = InteractiveSimulationRunner(config)
        runner.run_simulation()
    except Exception as e:
        print(f"{ColoredOutput.RED}Fatal error: {str(e)}{ColoredOutput.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()