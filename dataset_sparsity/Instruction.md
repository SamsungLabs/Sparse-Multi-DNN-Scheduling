## Dataset
Download the datasets from the following links:

### DarkFace
1. Dowload data from [google drive](https://drive.google.com/file/d/1DuwSRvsYzDpOHdRYG5bk7E45IMDdp5pQ/view)
2. Set your datapath [here]()
3. Run the following code to install the dataset and dataloader builder
```
git clone https://github.com/cuiziteng/ICCV_MAET.git
cd ICCV_MAET
pip install mmcv-full==1.1.5
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```

## Models

### Sparse Zoo
Download sparse model from sparsezoo:
```
pip install sparsezoo
pip install onnx2torch
mkdir models
python download_sparse_models.py
```