# Simulation of CNN Accelerator

This folder contains: 1. Performance model of Sparse CNN accelerator (EyerissV2), 2. Training code/script of sparse CNN model with different sparsity patterns based on [SparseML](https://docs.neuralmagic.com/products/sparseml)

You will get latency CSV files of different CNN models with different sparsity under folder. You can eigher 1. Download from [here](https://drive.google.com/drive/folders/1GZ5QNeJY10ngnLkUr_A7yXX8u7NDRc28?usp=sharing) or 2. Following the procudure below step by step to get CSV files.

## Train Sparse CNN Models

We use [SparseML](https://docs.neuralmagic.com/products/sparseml) to get sparse CNN models prunned by different sparsity patterns. We are experimenting with four different models (ResNet50, SSD300, MobileNet-V1, VGG16) and three different sparsity patterns (random, N-M block and filter/channel). For example, to get sparse ResNet50 models, run the following command: 
```
cd sparseml
bash resnet50_prune_models_regression.sh
```

Then, you will get `***.pt` models under `recipes_models` folder.

## Run CNN Simulator 

After you got moels, e.g. `your_model.pt`. Assuming the root path of your dataset is `/your/dataset/path`, you can run simulation on CNN accelertor by
```
python sim_eyerissv2.py --dataset_root /your/dataset/path  --load_model_path /path/to/your_model.pt
```

A csv file will be generated under this folder.
