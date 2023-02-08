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
    # model = convert("./models/resnet_pruned_nonquant/deployment/model.onnx")
    # model = convert("./models/resnet_pruned_quant/model.onnx")
    model = torchvision.models.resnet50(pretrained=True)
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

  # Construct Dataset
  data_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
  ])
  # data_root = '/home/CORP/hongxiang.f/Documents/dataset_samples'
  data_root = args.dataset_root
  dataset_paths =  [data_root + '/darkface', data_root + '/coco', data_root + '/exdark', data_root + '/imagenet']


  model.eval()
  from torch.quantization import get_default_qconfig
  from torch.quantization.quantize_fx import prepare_fx, convert_fx
  # from torch.ao.quantization import QConfigMapping

  # Apply Quantization
  qconfig = get_default_qconfig("fbgemm")
  # qconfig_mapping = QConfigMapping().set_global(qconfig)
  qconfig_dict = {"": qconfig}
  test_dataset = datasets.ImageFolder(root=dataset_paths[1],
                                            transform=data_transform)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size, shuffle=True,
                                            num_workers=1)
  def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
      for image, target in data_loader:
          model(image)
  example_inputs = (next(iter(test_loader))[0]) # get an example input
  prepared_model = prepare_fx(model, qconfig_dict)  # fuse modules and insert observers
  # for dataset_path in dataset_paths:
  #   print ("Processing: ", dataset_path)
  #   layer_sparsity = [[] for i in range(len(intern_hooks))]
  #   network_sparsity = []
  #   test_dataset = datasets.ImageFolder(root=dataset_path,
  #                                             transform=data_transform)
  #   test_loader = torch.utils.data.DataLoader(test_dataset,
  #                                             batch_size=args.batch_size, shuffle=True,
  #                                             num_workers=1)
  calibrate(prepared_model, test_loader)  # run calibration on sample data
  # print (prepared_model)
  model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
  # print (quantized_model)
  # sys.exit(0)

  # Insert hooks to collect intermediate results and calculate sparsity
  target_module_list = [torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d]
  # target_module_list = [torch.QuantizedConvReLU2d]
  model, intern_hooks = Stat_Collector.insert_hook(model, target_module_list)
  # sys.exit(0)
  model.cpu()
  model.eval()
  input_sparsities = [0, 1]
  means_sparse = [[] for i in range(len(intern_hooks))] 
  stddev_sparse = [[] for i in range(len(intern_hooks))] 

  # Run different dataset and record sparsity
  with torch.no_grad():
    layer_sparsities = []
    network_sparsities = []
    for dataset_path in dataset_paths:
      print ("Processing: ", dataset_path)
      layer_sparsity = [[] for i in range(len(intern_hooks))]
      network_sparsity = []
      test_dataset = datasets.ImageFolder(root=dataset_path,
                                                transform=data_transform)
      test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=1)
      for x, y in test_loader:
        x.cpu()
        output = model(x)
        sum_sparsity = 0.0 # Calculate network sparsity per sample
        for i, hook in enumerate(intern_hooks):
          sparsity = (hook.sparsity)
          sum_sparsity += sparsity
          layer_sparsity[i].append(sparsity)
        network_sparsity.append(sum_sparsity/len(intern_hooks))
      
      layer_sparsities.append(layer_sparsity)
      network_sparsities.append(network_sparsity)
 

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
  interdata_layer_sparsities = [[] for i in range(len(intern_hooks))]
  for i in range(len(intern_hooks)):
    for j in range(len(dataset_paths)):
      interdata_layer_sparsities[i] += layer_sparsities[j][i]
    # print ("Num of Samples:", len(interdata_layer_sparsities[i]))
    plt.hist(interdata_layer_sparsities[i], density=True)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    # plt.show()  
  plt.savefig("/homes/hf17/transfer/" + args.model_name + "_quantconv.png")


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