# Markov Chains

A collection of Python scripts for working with continuous-time (CTMC) and
discrete-time (DTMC) Markov chains. Each file handles a specific aspect of
Markov chain analysis, from simulation to spectral decomposition.

## Files

- **generate_chains.py**
Simulation utilities for CTMCs and DTMCs. `simulate_ctmc` runs a CTMC forward
in time by sampling exponential holding times and jumping between states using
probabilities derived from the generator matrix. `simulate_dtmc` steps through
a DTMC by sampling from each row of the transition matrix.

- **decompose_generator.py**
Decomposes a generator matrix Q into its holding rates (the diagonal of -Q)
and the embedded discrete-time Markov chain, which describes jump probabilities
between states independent of time.

- **generator_to_transition.py**
Converts a generator matrix Q into a transition probability matrix P(t) = exp(Q * t)
using the matrix exponential. Also provides `validate_generator` and
`validate_transition` to check whether a matrix satisfies the row-sum conditions
for a generator or stochastic matrix.

- **stationary_dist.py**
Computes the stationary distribution of a CTMC by finding the left eigenvector
of Q corresponding to eigenvalue 0 and normalizing it to sum to 1.

- **schur_decomp.py**
Computes the real Schur decomposition of a transition matrix. Useful for
analyzing the spectral structure of a DTMC, particularly when eigenvalues are
complex-valued.

## Requirements
Install dependencies with 
```python
pip install -r requirements.txt
```
