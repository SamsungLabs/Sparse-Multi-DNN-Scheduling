import argparse
import sys
import numpy as np
from data_util import cal_sparsity_tensor, create_sparse_tensor, Stat_Collector
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from onnx2torch import convert

def collect_sparse(args):
  # Define or Load Models
  img_size = 224
  if (args.model_name == "ssdlite"):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    img_size = 320
  elif (args.model_name == "ssd"):
    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model = convert("./models/ssd_prunned/deployment/model.onnx")
    img_size = 300
  elif (args.model_name == "resnet"):
    model = convert("./models/resnet_pruned_nonquant/deployment/model.onnx")
  elif (args.model_name == "mobilenetv1"):
    model = convert("./models/mobilenetv1_prunned/deployment/model.onnx")
  elif (args.model_name == "mobilenetv2"):
    model = convert("./models/mobilenetv2/deployment/model.onnx")
  elif (args.model_name == "vgg"):
    model = convert("./models/vgg16_prunned/deployment/model.onnx")
  elif (args.model_name == "inceptionv3"):
    model = convert("./models/inceptionv3_prunned/deployment/model.onnx")
  else:
    raise NameError('The model currently not supported')
  
  model.eval()

  target_module_list = [nn.ReLU]
  model, intern_hooks = Stat_Collector.insert_hook(model, target_module_list)
  input_sparsities = [0.1, 0.9]
  means_sparse = [[] for i in range(len(intern_hooks))] 
  stddev_sparse = [[] for i in range(len(intern_hooks))]
  with torch.no_grad():
    for input_sparsity in input_sparsities:
      sparsities = [[] for i in range(len(intern_hooks))] 
      for iter in range(args.num_samples):
        x = create_sparse_tensor((1, 3, img_size, img_size), input_sparsity)
        # print ("input sparsity:", cal_sparsity_tensor(x))
        output = model(x)
        for i, hook in enumerate(intern_hooks):
          sparsities[i].append(hook.sparsity)
      for i, sparsity in enumerate(sparsities):
        # print (i, "th layer with mean ", np.mean(sparsity), " and variance ", np.var(sparsity), np.std(sparsity))
        means_sparse[i].append(np.mean(sparsity))
        stddev_sparse[i].append(np.std(sparsity))
        # print (sparsity)
    for i, mean_sparse in enumerate(means_sparse):
      dif = np.abs(np.max(mean_sparse) - np.min(mean_sparse))
      print (i, "th layer, largest mean diff is ", dif)

if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset_name", default="darkface", type=str, help="The name of dataset to collect sparsity", choices=["mscoco", "darkface", "imagenet"])
  parser.add_argument("--model_name", default="inceptionv3", type=str, help="The name of model to collect sparsity", choices=["ssd", "ssdlite", "resnet", "mobilenet", "vgg", "inceptionv3"])
  parser.add_argument("--num_samples", default=50, type=int, help="The number of samples from the dataset to calculate sparsity")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")


  args = parser.parse_args()

  collect_sparse(args)