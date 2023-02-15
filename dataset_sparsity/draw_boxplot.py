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
from tqdm import tqdm

def collect_sparse(args):
  # Define or Load Models
  img_size = 224
  target_module_list = [nn.Conv2d, nn.Linear]
  data_root = args.dataset_root
  dataset_paths =  [data_root + '/darkface', data_root + '/coco', data_root + '/exdark', data_root + '/imagenet']
  fig, axs = plt.subplots(ncols=len(args.model_names), figsize=(13,5))
  colors = ['pink', 'lightblue', 'lightgreen', 'orange']
  models_str = ''
  for k in range(len(args.model_names)):
    model_name = args.model_names[k]
    models_str = models_str + model_name + '_'
    # Get models
    if (model_name == "ssdlite"):
      model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
      img_size = 320
    elif (model_name == "ssd"):
      model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
      # model = convert("./models/ssd_prunned/deployment/model.onnx")
      img_size = 300
    elif (model_name == "resnet"):
      model = convert("./models/resnet_pruned_nonquant/deployment/model.onnx")
      # model = torchvision.models.resnet50(pretrained=True)
    elif (model_name == "inceptionv3"):
      # model = convert("./models/mobilenetv1_prunned/deployment/model.onnx")
      target_module_list = [nn.Conv2d] # One Linear is not used
      img_size = 299
      model = torchvision.models.inception_v3(pretrained=True)
    elif (model_name == "mobilenetv2"):
      # model = convert("./models/mobilenetv2/deployment/model.onnx")
      model = torchvision.models.mobilenet_v2(pretrained=True)
    elif (model_name == "vgg"):
      # model = convert("./models/vgg16_prunned/deployment/model.onnx")
      model = torchvision.models.vgg16(pretrained=True)
    elif (model_name == "googlenet"):
      # model = convert("./models/googlenet_prunned/deployment/model.onnx")
      model = torchvision.models.googlenet(pretrained=True)
    else:
      raise NameError('The model currently not supported')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Construct Dataset
    data_transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # Insert hooks to collect intermediate results and calculate sparsity
    model, intern_hooks = Stat_Collector.insert_hook(model, target_module_list)
    input_sparsities = [0, 1]
    means_sparse = [[] for i in range(len(intern_hooks))] 
    stddev_sparse = [[] for i in range(len(intern_hooks))] 

    # Run different dataset and record sparsity
    with torch.no_grad():
      layer_sparsities = []
      network_valid_macs = []
      mac_propotions = []
      for dataset_path in dataset_paths:
        print ("Processing: ", dataset_path)
        layer_sparsity = [[] for i in range(len(intern_hooks))]
        mac_propotion = [[] for i in range(len(intern_hooks))]
        network_valid_mac = []
        test_dataset = datasets.ImageFolder(root=dataset_path,
                                                  transform=data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=1)
        with tqdm(total=len(test_loader)) as pbar:
          for x, y in test_loader:
            pbar.update(1)
            x = x.to(device)
            output = model(x)
            sum_sparsity = 0.0 # Calculate network sparsity per sample
            sum_v_mac = 0
            sum_t_mac = 0
            v_mac = []
            for i, hook in enumerate(intern_hooks):
              # print ("hook", hook.sparsity)
              if (isinstance(hook.sparsity, float)): continue # Layer not running
              sparsity = (hook.sparsity.cpu())
              v_mac.append(hook.v_mac.cpu())
              sum_v_mac += hook.v_mac.cpu()
              sum_t_mac += hook.t_mac

              sum_sparsity += sparsity
              layer_sparsity[i].append(sparsity)
            for i in range(len(v_mac)):
              mac_propotion[i].append(v_mac[i]/sum_t_mac)
            # network_valid_mac.append(sum_sparsity/len(intern_hooks))
            network_valid_mac.append((sum_v_mac/sum_t_mac).numpy())
          # print (dataset_path, " ", layer_sparsity)
          # Remove empty list
          layer_sparsity = [s for s in layer_sparsity if s!=[]]
          layer_sparsities.append(layer_sparsity)
          mac_propotion = [m for m in mac_propotion if m!=[]]
          mac_propotions.append(mac_propotion)
          network_valid_macs.append(network_valid_mac)

    num_hook_layer = len(layer_sparsities[0])
    interdata_layer_sparsities = [[] for i in range(num_hook_layer)]
    for i in range(num_hook_layer):
      for j in range(len(dataset_paths)):
        interdata_layer_sparsities[i] += layer_sparsities[j][i]
    
    layer_index = list(range(1, 7)) # Start from begin
    # layer_index = list(range(-7, -1)) # Start from end
    # test = [interdata_layer_sparsities[l] for l in layer_index]
    # print (len(test[0]))
    #  Draw layer sparsity
    if (len(args.model_names) > 1): ax = axs[k]
    else: ax = axs
    ax.boxplot([interdata_layer_sparsities[l] for l in layer_index], whis=[0,100], positions=layer_index, patch_artist=True,
                    boxprops=dict(facecolor=colors[k],color=colors[k]), medianprops=dict(color='black'))
    ax.set_ylabel('Sparsity', fontsize=16)
    ax.set_xlabel('Layer Index', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    # Calculate the percentage relative range

    cross_network_valid_macs = []
    for i in range(len(dataset_paths)):
      cross_network_valid_macs += network_valid_macs[i]
    mean = np.mean(cross_network_valid_macs)
    max = np.max(cross_network_valid_macs)
    min = np.min(cross_network_valid_macs)
    prr = (max - min)/mean
    print ("Percentage Relative Range of ", model_name, ":", prr)


  fig.savefig(args.figs_path + models_str + "layer_sparsity.pdf")
  plt.close()


  # Analyse the sparsity distribution

  #  Draw intra (in) dataset network sparsity distribution

  # fig,axs = plt.subplots(4, figsize=(9,20))
  # for i in range(len(dataset_paths)):
  #   axs[i].hist(network_valid_macs[i], density=True)  # density=False would make counts
  #   axs[i].set_ylabel('Probability')
  #   axs[i].set_xlabel(dataset_paths[i][-4:])
  # fig.savefig(args.figs_path + model_name + "_intra_network_valid_mac.png")
  # plt.close()

  


if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", default="darkface", type=str, help="The name of dataset to collect sparsity", choices=["mscoco", "darkface", "imagenet"])
  parser.add_argument("--model_names", '--names-list', nargs='+', default="googlenet", type=str, help="The name of model to collect sparsity", choices=["ssd", "ssdlite", "resnet", "inceptionv3", "mobilenetv2", "vgg", "googlenet"])
  parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  parser.add_argument("--figs_path", default="/path/to/your/directory", type=str, help="The path to all saved figures/images")
  parser.add_argument("--num_samples", default=50, type=int, help="The number of samples from the dataset to calculate sparsity")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

  args = parser.parse_args()

  collect_sparse(args)