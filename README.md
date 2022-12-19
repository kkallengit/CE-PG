# Source code for CE-PG
This repo includes the implementation of CE-PG Algorithm described in the paper **Cross Entropy Regularized Policy Gradient for Multi-Robot Non-Adversarial Moving Target Search**


## Description
The source code includes the followings: 
* `CEPG.py`: core code.
* `env.py`: simulation environment.
* `func.py`: target state estimation module.
* `network.py`: 
* `PTB_update.py`: 

## Core ideas for code
- This code deals with the multi-robot efficient search (MuRES) for a non-adversarial moving target problem from the multi-agent reinforcement learning (MARL) perspective. 
- This code includes the estimate of the target motion dynamics, the PTB (Probabilistic Target Belief) update, and the cross entropy regularized policy gradient searching method. 
- This code consists of two main modules, namely *Online Decentralized Execution* and *Offline Centralized Training*, which are described in detail in the paper with pseudo-code.
- The loss function of the algorithm can be described as:

$$\tilde{J}(\boldsymbol{\theta}_i)=\beta_i J(\pi(\boldsymbol{\theta}_i))+\frac{1-\beta_i}{N-1}\sum_{j\neq i}\mathcal{H}\big(\pi(\boldsymbol{\theta}_j),\pi(\boldsymbol{\theta}_i)\big)$$

## Dependencies
- Python 3.7+
- Numpy
- Pandas
- Torch
