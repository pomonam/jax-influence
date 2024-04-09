<div align="center">    

## JAX-Influence

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)

</div>

JAX-Influence is a JAX implementation of influence functions, a classical technique from robust statistics that
estimates the impact of removing a single training data point on a model's learned parameters. This repository
complements the paper ["If Influence Functions are the Answer, Then What is the Question?"](https://arxiv.org/abs/2209.05364).

The repository aims to provide a simple and minimal implementation of influence functions in JAX. For those interested in 
implementations in other frameworks, a PyTorch version is available [here](https://github.com/alstonlo/torch-influence), and 
a PyTorch EK-FAC implementation can be found [here](https://github.com/pomonam/kronfluence).

## Installation

To install JAX-Influence, you can use pip to install from the source:

```bash
git clone https://github.com/pomonam/jax-influence
 
cd jax-influence
pip install -e .   
pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html' # Replace `jax_gpu` with `jax_cpu` if you wish to install the CPU version.
 ```

## Contributors

- [Juhan Bae](https://www.juhanbae.com/)
- [Nathan Ng](https://nng555.github.io/)
- [Alston Lo](https://alstonlo.github.io/)
