# Ohm Principle Simulation
This repository contains the Python code for the Q-learning simulation described in the manuscript "The Ohm Principle: A Testable Framework for Generative Change" submitted to *Entropy* (August 31, 2025). The code implements a Q-learning agent in a deterministic FrozenLake environment to test the Volition Hypothesis, with mutual information and Sample Entropy analyses.

## Files
- `ohm_principle_simulation.py`: Main simulation code, generating results and Figures 5.1 and 5.2.
- `figure_5.1.png`: Mean reward per episode plot.
- `figure_5.2.png`: Mean Sample Entropy per episode plot.
- `simulation_results.csv`: Summary statistics (episodes, MI, Sample Entropy).

## Requirements
- Python 3.8+
- Libraries: numpy, scipy, scikit-learn, matplotlib
- Install via: `pip install numpy scipy scikit-learn matplotlib`

## Usage
Run `ohm_principle_simulation.py` to replicate the simulation. Outputs include:
- Mean episodes to Omega, mutual information, and Sample Entropy for Fixed, Random, and Volitional Agents.
- Figures 5.1 and 5.2 as PNG files.
- Summary statistics in `simulation_results.csv`.

## Citation
Ohm, L.S. (2025). The Ohm Principle: A Testable Framework for Generative Change. *Entropy* (under review).
