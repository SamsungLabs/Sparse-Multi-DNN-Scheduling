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
import matplotlib.pyplot as plt

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
    model = convert("./models/vgg16_prunned_c/deployment/model.onnx")
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
  # data_root = '/home/CORP/hongxiang.f/Documents/dataset_samples'
  data_root = args.dataset_root
  dataset_paths =  [data_root + '/darkface', data_root + '/imagenet']

  # Insert hooks to collect intermediate results and calculate sparsity
  # target_module_list = [nn.Conv2d]
  target_module_list = [nn.ReLU]
  model, intern_hooks = Stat_Collector.insert_hook(model, target_module_list)
  input_sparsities = [0, 1]
  means_sparse = [[] for i in range(len(intern_hooks))] 
  stddev_sparse = [[] for i in range(len(intern_hooks))] 

  # Run different dataset and record sparsity
  with torch.no_grad():
    layer_features = []
    network_sparsities = []
    for dataset_path in dataset_paths:
      sample = 0
      print ("Processing: ", dataset_path)
      layer_feature = [[] for i in range(len(intern_hooks))]
      test_dataset = datasets.ImageFolder(root=dataset_path,
                                                transform=data_transform)
      test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=1)
      for x, y in test_loader:
        sample += 1
        if (sample > args.num_samples): break
        output = model(x)
        for i, hook in enumerate(intern_hooks):
          layer_feature[i] += (list(hook.out_features.flatten().numpy()))
      
      layer_features.append(layer_feature)
 

  # Analyse the sparsity distribution

  #  Draw intra (in) dataset network sparsity distribution

  # for i in range(len(dataset_paths)):
  #   plt.hist(network_sparsities[i], density=True)  # density=False would make counts
  #   plt.ylabel('Probability')
  #   plt.xlabel('Data')
  #   plt.show()  

  #  Draw inter (cross) dataset network sparsity distribution

  # cross_network_sparsities = []
  # for i in range(len(dataset_paths)):
  #   cross_network_sparsities += network_sparsities[i]
  # plt.hist(cross_network_sparsities, density=True)  # density=False would make counts
  # plt.ylabel('Probability')
  # plt.xlabel('Data')
  # plt.show()  

  #  Draw inter (cross) dataset layer sparsity distribution
  
  interdata_layer_features = [[] for i in range(len(intern_hooks))]
  colors = ['gold', 'red']
  for i in range(len(intern_hooks)):
    plt.figure()
    for j in range(len(dataset_paths)):
      # print (layer_features[j][i])
      # interdata_layer_features[i] += layer_features[j][i]
      # print ("Num of Samples:", len(interdata_layer_sparsities[i]))
      plt.hist(layer_features[j][i], density=True, bins=100, color=colors[j], range=(0.0, 1.0))  # density=False would make counts
      plt.ylabel('Probability')
      plt.xlabel('Data')
      # plt.show() 
    plt.savefig("/homes/hf17/transfer/" + args.model_name + str(i) +"_dist_new.png")
    plt.close()

  #  Draw inter (cross) dataset layer sparsity distribution
  
  # interdata_layer_features = [[] for i in range(len(intern_hooks))]
  # for i in range(len(intern_hooks)):
  #   for j in range(len(dataset_paths)):
  #     # print (layer_features[j][i])
  #     interdata_layer_features[i] += layer_features[j][i]
  #   # print ("Num of Samples:", len(interdata_layer_sparsities[i]))
  #   plt.figure()
  #   plt.hist(interdata_layer_features[i], density=True, bins=10)  # density=False would make counts
  #   plt.ylabel('Probability')
  #   plt.xlabel('Data')
  #   # plt.show() 
  #   plt.savefig("/homes/hf17/transfer/" + args.model_name + str(i) +"_dist.png")
  #   plt.close()


if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", default="darkface", type=str, help="The name of dataset to collect sparsity", choices=["mscoco", "darkface", "imagenet"])
  parser.add_argument("--model_name", default="inceptionv3", type=str, help="The name of model to collect sparsity", choices=["ssd", "ssdlite", "resnet", "mobilenetv1", "mobilenetv2", "vgg", "inceptionv3"])
  parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  parser.add_argument("--num_samples", default=50, type=int, help="The number of samples from the dataset to calculate sparsity")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

  args = parser.parse_args()

  collect_sparse(args)