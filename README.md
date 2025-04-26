# Heterogeneous Firms with Deep Learning Approximation

This repository contains code to solve and simulate heterogeneous-firm investment models under uncertainty, using TensorFlow-based deep learning techniques to approximate policy functions.

## Project Overview

We develop two dynamic models of firm behavior:
- `DL_firm.py`: A model where firms face idiosyncratic persistent productivity shocks.
- `DL_firm_multi_shock.py`: An extended model with persistent productivity shocks, transitory shocks, and capital quality shocks.

Both models:
- Use neural networks to approximate firms' optimal investment (capital accumulation) decisions.
- Solve for the general equilibrium wage rate through a bisection algorithm.
- Simulate large economies (hundreds of thousands of firms) over long time horizons.
- Monitor convergence through Euler equation residuals.
- Visualize policy functions and aggregate macroeconomic outcomes.

## File Descriptions

| File | Description |
| :--- | :---------- |
| `DL_firm.py` | Baseline model with idiosyncratic productivity shock, solving the investment problem using deep learning. |
| `DL_firm_multi_shock.py` | Extended model including persistent, transitory, and capital quality shocks. |
| `DL_firm_notebook.ipynb` | Jupyter notebook for running, analyzing, and visualizing simulations interactively. |

## Key Features

- TensorFlow 2.x implementation of policy function approximation.
- Bisection method for equilibrium computation.
- Extensive use of GPU acceleration (if available).
- Visualization of training error, capital policy functions, and macro aggregates like output and consumption.
- Robust to explosive solutions through model re-initialization strategies.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm

Install dependencies via:
```bash
pip install tensorflow numpy matplotlib tqdm
