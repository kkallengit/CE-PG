# Source code for CE-PG
The repository includes the implementation of CE-PG Algorithm described in the paper **Cross Entropy Regularized Policy Gradient for Multi-Robot Non-Adversarial Moving Target Search**, and the benchmarks, namely MARL methods used for comparison.


## Description
The source code includes the followings: 
* `CEPG.py`: core code of CE-PG Algorithm. 
* `env.py`: two canonical multi-robot effective test environments, namely OFFICE and MUSEUM.
* `func.py`: tool functions used throughout the project.
* `network.py`: specific input and output, amount of layers, number of nodes, and activation function design for neural networks.
* `PTB_update.py`: PTB update function whose output is the composition of robot *state*.
* `CE-PG demonstrative video.mp4`: a demonstrative video which demonstrates the framework of CE-PG, the physical experimental procedure and results, and the summary of the contributions of this paper.

## Core ideas for code
- This code deals with the multi-robot efficient search (MuRES) for a non-adversarial moving target problem from the multi-agent reinforcement learning (MARL) perspective. 
- This code includes the estimate of the target motion dynamics, the PTB (Probabilistic Target Belief) update, and the cross entropy regularized policy gradient searching method. 
- This code consists of two main modules, namely *Online Decentralized Execution* and *Offline Centralized Training*, which are described in detail in the paper with pseudo-code.
- The loss function of the algorithm can be described as:

$$\tilde{J}(\boldsymbol{\theta}_i)=\beta_i J(\pi(\boldsymbol{\theta}_i))+\frac{1-\beta_i}{N-1}\sum_{j\neq i}\mathcal{H}\big(\pi(\boldsymbol{\theta}_j),\pi(\boldsymbol{\theta}_i)\big)$$

where $J(\pi(\boldsymbol{\theta}_i))$ represents the individual expected return of robot $i$ and $\mathcal{H}\big(\pi(\boldsymbol{\theta}_j),\pi(\boldsymbol{\theta}_i)\big)$ means robot $i$'s expected cross entropy which disperses the robots from each other.
## Dependencies
- Python 3.7+
- Numpy
- Pandas
- Torch
