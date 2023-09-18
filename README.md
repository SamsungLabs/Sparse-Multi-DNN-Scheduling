# Sparse Multi-DNN Scheduling
Open-source artifacts and codes of our MICRO'23 paper titled "Sparse-DySta: Sparsity-Aware Dynamic and Static Scheduling for Sparse Multi-DNN Workloads". This repo mainly contains:
- Sparse Multi-DNN **Benchmark**: Collections of diverse sparsified ([SparseML](https://neuralmagic.com/sparseml/)) Convolutional NNs (CNNs) and Attention-based NNs (AttNNs) from three distinct applications, namely visual perception, personal assistant, and hand tracking.
- Simulation-based Evaluation **Infrastructure**: Seamless integration with `PyTorch` to evaluate the performance of different multi-DNN scheduling approaches.
- Prototype of our Static and Dynamic **Scheduler**: FPGA-based implementation to obtain resource estimation of hardware resources.


## 1. Structure

```
.
├── README.md
├── dataset_sparsity   # Code for motivation part, profiling MAC counts and latency across different sparse accelerators
├── dysta_scheduler    # Code for evaluating different scheduling approaches
    ├── csv_files      # CSV files of runtime (latency) information, which are extracted by simulating performance models on CNN/AttNNs.
    ├── script         # Script for reproducing results
├── requirements.txt     
├── hw_design          # Verilog code of hardware scheduler
|                      # The complete Vivado projects are already provided in Docker image under /workspace/hw_dysta and  hw_monitor_eyeriss
└── hw_simulator       # Code for obtaining CSV files  
```


## 2. Installation

You can install the environment through either Docker or Conda. We recommend you to use the Docker image since it contains all the necessary CSV runtime information files. Otherwise you will need to re-generate, which may take some time.

### 2.1 Docker

Download Docker image by the following steps: 
```
docker pull hxfan/spar-dysta-micro23:ae
```
Follow Docker [Tutorial](https://docs.docker.com/get-started/) to create Container. All the dependencies have been installed in the provided Docker image. To setup docker container:
```
sudo docker run -it -d --name spar-dysta-container --gpus all  hxfan/spar-dysta-micro23:ae /bin/bash # Create container
sudo docker exec -it spar-dysta-container /bin/bash
# Inside container
cd /workspace/dysta-sparse
```
Pls note the complete Vivado projects have been provided in `/workspace/hw_dysta` and  `hw_monitor_eyeriss`.
### 2.2 Conda [Optional]
Install `Anaconda` environment with `Python 3.8.12` and `CUDA-11.3`, running the following commands:
```
# create environment and activate
conda create -n dysta-sparse python=3.8.12 -y
conda activate dysta-sparse
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Other dependencies:
```
pip install matplotlib==3.6.3 pandas scipy scikit-learn brokenaxes onnx2torch sparseml
```

## 3. Reproducibility/Artifact Evaluation

We provide scripts to generate all the figures and tables. Detailed instructions on how to run these scripts are provided in [INST_RPRODUCE.md](./INST_RPRODUCE.md).

## 4. MLCommons CM Interface (Artifacts Reusable)

`CM` scripts are provided at [here](https://github.com/ctuning/cm-reproduce-research-projects). You will need to install `CM` and add the corresponding `CM` scripts using this [guidance](https://github.com/ctuning/cm-reproduce-research-projects#readme). The following commands are used to install dependencies and run experiments.
```
# Install dependencies
cm run script "reproduce paper m2023 5 _install_deps"
# Run experiments
cm run script "reproduce paper m2023 5 _run"
```

## Citation 

If you found it helpful, pls cite us using:


``` 
@inproceedings{fan2023sparse-dysta,
  title={{Sparse-DySta: Sparsity-Aware Dynamic and Static Scheduling for Sparse Multi-DNN Workloads}},
  author={Fan, Hongxiang and Venieris, Stylianos I. and Kouris, Alexandros and  Lane, Nicholas D.},
  booktitle={56th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2023}
}

```
