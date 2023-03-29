

# Insert hook

# Define simulator for each layer

import argparse
import sys
import numpy as np
from utils_sim import insert_hook_simulator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
import logging
from pathlib import Path
import os
from tqdm.auto import tqdm

def run_sim(args):
  '''
  Iterate the dataset using the sparse models.
  Use hook to capture the intermedate results, then feed to Eyeriss-V2 simulator to get performance.
  '''
  # Get model name
  model_name = args.load_model_path.split('/')[-1].split('.')[0]
  # Load Sparse torch model
  #   Get input size
  data_root = args.dataset_root
  dataset_paths =  [data_root + '/darkface', data_root + '/exdark'] #data_root + '/imagenet']
  img_size = -1
  if any(name in model_name for name in ["resnet50", "vgg16", "mobilenet"]):
    img_size = 224
    dataset_paths = ['imagenette', data_root + '/coco'] + dataset_paths
  elif ("ssd" in model_name):
    img_size = 300
    dataset_paths = ['voc'] + dataset_paths
  else:
    raise NameError('The model currently not supported')
  #   Load from path
  model = torch.load(args.load_model_path)
  model.eval()
  print(model)
  
  # Get CSV writer
  csv_path = Path(model_name + '.csv')
  csv_file = csv_path.open('w')
  csv_file.write('sim_lat,overall-sparsity,batch-indx,layer-indx\n') # We keep the naming consistent with Sanger
  
  # Construct Dataset
  data_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
  ])

  # Apply Quantization, TO-DO

  # from torch.quantization import get_default_qconfig
  # from torch.quantization.quantize_fx import prepare_fx, convert_fx

  # qconfig = get_default_qconfig("fbgemm")
  # qconfig_dict = {"": qconfig}
  # test_dataset = datasets.ImageFolder(root=dataset_paths[1],
  #                                           transform=data_transform)
  # test_loader = torch.utils.data.DataLoader(test_dataset,
  #                                           batch_size=args.batch_size, shuffle=True,
  #                                           num_workers=1)
  # def calibrate(model, data_loader):
  #   model.eval()
  #   with torch.no_grad():
  #     for image, target in data_loader:
  #         model(image)
  # example_inputs = (next(iter(test_loader))[0]) # get an example input
  # prepared_model = prepare_fx(model, qconfig_dict)  # fuse modules and insert observers
  # calibrate(prepared_model, test_loader)  # run calibration on sample data
  # model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model


  # Insert hooks and construtc eyeriss simulator for each layer
  target_module_list = [nn.Conv2d, nn.Linear]
  model, intern_hooks = insert_hook_simulator(model, target_module_list)

  model.cpu()
  input_sparsities = [0, 1]
  means_sparse = [[] for i in range(len(intern_hooks))] 
  stddev_sparse = [[] for i in range(len(intern_hooks))] 

  # Iterate through dataset
  cur_batch_num = 0
  with torch.no_grad():
    layer_sparsities = []
    network_sparsities = []
    for dataset_path in dataset_paths:
      print ("Processing: ", dataset_path)
      layer_sparsity = [[] for i in range(len(intern_hooks))]
      network_sparsity = []
      if 'imagenette' in dataset_path:
        test_dataset = ImagenetteDataset(
            train=False, dataset_size=ImagenetteSize.s320, image_size=img_size
        )
        test_loader = DataLoader(
            test_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=8
        )
      elif 'voc' in dataset_path:
        from sparseml.pytorch.datasets import VOCDetectionDataset
        from sparseml.pytorch.utils import get_default_boxes_300
        from sparseml.pytorch.datasets import ssd_collate_fn
        default_boxes = get_default_boxes_300("voc")
        test_dataset = VOCDetectionDataset(
            train=False, preprocessing_type="ssd", default_boxes=default_boxes
        )
        test_loader = DataLoader(
            test_dataset,
            args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=12,
            collate_fn=ssd_collate_fn,
        )
      else: # Out of distibution dataset
        test_dataset = datasets.ImageFolder(root=dataset_path,
                                                  transform=data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=1)   
                                              
      for step, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        x.cpu()
        output = model(x)
        for layer_indx, hook in enumerate(intern_hooks):
          csv_file.write(str(hook.estimated_cycle/(args.clock_freq * 1000000)) + ',' + str(hook.iact_sparsity) + ','+str(cur_batch_num)+','+str(layer_indx)+'\n')
        cur_batch_num += 1

if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--figs_path", default="/path/to/your/directory", type=str, help="The path to all saved figures/images")
  parser.add_argument("--load_model_path", default="./sparseml/recipes_models/resnet50_unstructured_sparse80.pt", type=str, help="The path where you load the pre-trianed sparse model")
  parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
  parser.add_argument("--clock_freq", default=200, type=float, help="Clock frequency (in MHz) of EyerissV2")

  # Verbosity / Logging
  parser.add_argument("--debug", action="store_true")

  args = parser.parse_args()

  # Logging setup
  if args.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  run_sim(args)