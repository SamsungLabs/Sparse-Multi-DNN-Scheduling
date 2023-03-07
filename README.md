# dysta-sparse
Code for project "Sparsity-Aware Dynamic and Static Scheduling for Sparse Multi-DNNs Workloads".


## Installation

Install `Anaconda` environment with `Python 3.8.12` and `CUDA-11.3`, running the following commands:
```
# create environment and activate
conda create -n dysta-sparse python=3.8.12 -y
conda activate dysta-sparse
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Other dependencies:
```
pip install matplotlib pandas
```