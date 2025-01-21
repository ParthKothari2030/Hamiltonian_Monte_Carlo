# Hamiltonian Monte Carlo


## Overview

This repository contains two Jupyter notebooks demonstrating the implementation of Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings MCMC (MH-MCMC) for parameter estimation of a polynomial model:

$$
y = a_0 + a_1 \cdot x + a_2 \cdot (x^2),
$$

where the parameters $a_0$, $a_1$, and $a_2$ are inferred by fitting a given dataset. The dataset contains three columns:
1. **x**: Independent variable.
2. **y**: Dependent variable.
3. **sigma**: Error in $y$.

### Task Description
1. **Likelihood Function**:
   - The likelihood is defined assuming a Gaussian error model with a diagonal covariance matrix. The diagonal elements are given by $\sigma^2$ (third column of the data file).

   $$
   \mathcal{L}(a_0, a_1, a_2) \propto \exp\left(-\frac{1}{2} \sum_{i=1}^{N} \frac{(y_i - (a_0 + a_1 x_i + a_2 x_i^2))^2}{\sigma_i^2}\right).
   $$

2. **Priors**:
   - Uniform priors are assumed for the parameters within the following ranges:
     - $a_0 \in [500, 2000]$
     - $a_1 \in [0, 10]$
     - $a_2 \in [0, 5]$

3. **Implementation**:
   - The problem is solved using HMC and MH-MCMC methods. 
   - The notebooks demonstrate the inference process, visualize the results, and provide corner plots of the parameter posterior distributions.

## Notebooks

### 1. `HMCfromscratch.ipynb`
- **Purpose**: Implements the Hamiltonian Monte Carlo algorithm using only NumPy for demonstration purposes.
- **Key Features**:
  - Manual implementation of HMC.
  - Derivation of the gradient of the log-likelihood function.
  - Sampling from the posterior distribution of $a_0$, $a_1$, and $a_2$.
  - Visualization of results using corner plots.
  - Code for visualizing the chains in the parameter space

### 2. `HMCusingtensorflow.ipynb`
- **Purpose**: Demonstrates using TensorFlow for efficient HMC sampling and compares it with MH-MCMC using the `emcee` library.
- **Key Features**:
  - Implementation of HMC using TensorFlow.
  - Step-by-step guide to configuring the HMC algorithm.
  - Demonstration of MH-MCMC using the `emcee` library for the same problem.
  - Comparison of the results obtained from both methods.

## Requirements

To run the notebooks, install the following Python packages:

```bash
pip install numpy scipy matplotlib corner tensorflow tensorflow-probability emcee
```

## File Structure

- `HMCfromscratch.ipynb`: Implementation of HMC using NumPy.
- `HMCusingtensorflow.ipynb`: Implementation of HMC using TensorFlow and MH-MCMC using `emcee`.
- `data.txt`: Input data file containing three columns (x, y, sigma).
- `MHMCMC_output.npy`: Data for MHMCMC run using `emcee`, the first second and third column represent the estimated parameters $a_0$ , $a_1$ and $a_2$

## How to Use

1. Clone the repository:
   ```bash
   git clone [<repository_url>](https://github.com/ParthKothari2030/Hamiltonian_Monte_Carlo.git)
   cd <repository_name>
   ```

2. Open the notebooks in Jupyter:
   ```bash
   jupyter notebook
   ```

3. Follow the step-by-step instructions in each notebook to understand the implementation.

## Results

Both notebooks generate posterior distributions of the parameters $a_0$, $a_1$, and $a_2$. The results are visualized using corner plots, which show:
- Marginal distributions for each parameter.
- Pairwise correlations between parameters.

The TensorFlow-based implementation demonstrates improved efficiency and ease of use compared to the manual implementation.

## References
- [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
- [emcee: The MCMC Hammer](https://emcee.readthedocs.io/en/stable/)
- [TensorFlow Probability](https://www.tensorflow.org/probability)

## Contact
For any questions or issues, please open an issue in the repository or contact the author.
