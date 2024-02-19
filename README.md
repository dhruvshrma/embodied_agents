# Embodied Agents 

Code for the presentation on [Embodied Agent-based Modeling](https://dhruv-sharma.ovh/post/talk-ccs-2023/presentation_ccs2023.pdf)

## Code Structure 

The code is structured as follows:

- `agents`: Contains definitions for the different types of agents used. 
    There are two types for the purposes of the simulation: `SimpleAgent` and `MediatingAgent`. 
- `environments`: Contains the definition for the environments for the interaction of the agents. The agents are all graph-based, and three types of graphs typically used in network science are used: `star`, `scale-free`, and `small-world`. 
- `interactions`: Contains the definition for the possible interactions between the agents. For social dynamic interactions, we can have `MajorityRule` or a more general `VoterModel`. For the presentation, code was present in `DialogueSimulation`. 
- `personas`: Contains the definition and construction of the different personas used in the simulations. 
- `simulation`: Contains the code for running the simulations. Specific "factory" classes for generating and managing agents. 
- `utils`: Various utilities, especially logging. 

We also provide a series of tests within the `tests` folder. These can be run using `pytest`. 

## TODOs 

- [ ] Add code for the calculations for similarity in opinions. 
- [ ] Add example notebook for showing a complete simulation. 

