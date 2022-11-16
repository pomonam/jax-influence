<div align="center">    

# jax-influence

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)

</div>

jax-influence is a Jax implementation of influence functions, a classical
technique from robust statistics that estimates the effect of removing a single training data point on a model’s
learned parameters. The code is supplement to the paper [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/abs/2209.05364).

This library aims to be simple and minimal. Furthermore, the PyTorch implementation can be found at [here](https://github.com/alstonlo/torch-influence).

______________________________________________________________________

## Installation

Pip from source:

```bash
git clone https://github.com/pomonam/jax-influence
 
cd jax-influence
pip install -e .   
pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html' # Replace `jax_gpu` with `jax_cpu` if you wish to install the CPU version.
 ```

______________________________________________________________________

## Quickstart

### Overview

An end-to-end example can be found in `tests`. We will add more examples in the future, including PBRF computation.

______________________________________________________________________

## Contributors

- [Juhan Bae](https://www.juhanbae.com/)
- [Nathan Ng](https://scholar.google.com/citations?user=psuwztYAAAAJ&hl=en)
- [Alston Lo](https://github.com/alstonlo)
