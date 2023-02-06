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

  # Construct Dataset
  data_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
  ])
  data_root = '/mnt/ccnas2/bdp/hf17/MICRO/MICRO_23/dysta-sparse/dataset_sparsity/data_example'
  dataset_paths =  [data_root + '/darkface', data_root + '/mscoco', data_root + '/exdark']

  # Insert hooks to collect intermediate results and calculate sparsity
  target_module_list = [nn.ReLU]
  model, intern_hooks = Stat_Collector.insert_hook(model, target_module_list)
  input_sparsities = [0, 1]
  means_sparse = [[] for i in range(len(intern_hooks))] 
  stddev_sparse = [[] for i in range(len(intern_hooks))] 

  # Run different dataset and record sparsity
  with torch.no_grad():
    for dataset_path in dataset_paths:
      print ("Processing: ", dataset_path)
      sparsities = [[] for i in range(len(intern_hooks))]
      test_dataset = datasets.ImageFolder(root=dataset_path,
                                                transform=data_transform)
      test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=1)
      for x, y in test_loader:
        output = model(x)
        for i, hook in enumerate(intern_hooks):
          sparsities[i].append(hook.sparsity)
      for i, sparsity in enumerate(sparsities):
        means_sparse[i].append(np.mean(sparsity))
        stddev_sparse[i].append(np.std(sparsity))
    print ("mean list:", means_sparse)

  # Analyse the sparsity distribution
  for i, mean_sparse in enumerate(means_sparse):
    dif = np.abs(np.max(mean_sparse) - np.min(mean_sparse))
    print (i, "th layer, largest mean diff is ", dif)

if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", default="darkface", type=str, help="The name of dataset to collect sparsity", choices=["mscoco", "darkface", "imagenet"])
  parser.add_argument("--model_name", default="inceptionv3", type=str, help="The name of model to collect sparsity", choices=["ssd", "ssdlite", "resnet", "mobilenetv1", "mobilenetv2", "vgg", "inceptionv3"])
  parser.add_argument("--num_samples", default=50, type=int, help="The number of samples from the dataset to calculate sparsity")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

  args = parser.parse_args()

  collect_sparse(args)