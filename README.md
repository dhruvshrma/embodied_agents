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

### Instructions on how to run the code:

#### Using Conda

1. Install Conda or Miniconda if you haven't.

2. Clone this repository and navigate into it from your terminal.

```
git clone https://github.com/dhruvshrma/embodied-agents.git
cd embodied-agents
```
3. Create a new Conda environment using the provided `environment.yml` file.

```
conda env create -f environment.yml
```

4. Activate the newly created conda environment:

```
conda activate embodied-agents
```

5. Now, you can run the code.

#### Using pip (with a virtual environment)

1. Install Python if you haven't.

2. Clone this repository and navigate into it from your terminal.

```
git clone https://github.com/dhruvshrma/embodied-agents.git
cd embodied-agents
```
3. Create a new virtual environment (optional but recommended).

```
python -m venv env
```

4. Activate the virtual environment:

   On Windows:

   ```
   .\env\Scripts\activate
   ```

   On Unix or Linux:

   ```
   source env/bin/activate
   ```

5. Install the required dependencies from the `requirements.txt` file.

```
pip install -r requirements.txt
```

6. Now, you can run the code.

---
## TODOs 

- [ ] Add code for the calculations for similarity in opinions. 
- [X] Add example notebook for showing a complete simulation. 

