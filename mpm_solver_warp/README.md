# WARP MPM
The mpm solver is based on [warp-mpm](https://github.com/zeshunzong/warp-mpm). We modified it to allow gradient computation in the warp package. Specifically, we set `requires_grad=True` for the physical parameters in [mpm_solver_warp.py](./mpm_solver_warp.py) and replaced the in-place operation in [mpm_utils.py](./mpm_utils.py). The content below is the readme information from the original repo:

This MPM solver is implemented using Nvidia's WARP: https://nvidia.github.io/warp/

For details about MPM, please refer to the course on the Material Point Method: https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf


## Prerequisites

This codebase is tested using the environment with the following key packages:

- Ubuntu 20.04
- CUDA 11.6
- Python 3.9.13
- Warp 0.10.1

## Installation
```
pip install -r requirements.txt
```

## Examples
Sand: column collapse 
```
python run_sand.py
```

More coming.