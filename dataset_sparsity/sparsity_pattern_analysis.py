import argparse
import sys
import numpy as np
from data_util import cal_sparsity_tensor, create_sparse_tensor, Sparse_Pattern_Analyzer
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
  target_module_list = [nn.Linear, torch.nn.Conv2d]
  data_root = args.dataset_root
  dataset_paths =  [data_root + '/darkface', data_root + '/coco', data_root + '/exdark', data_root + '/imagenet']
  fig, axs = plt.subplots(ncols=len(args.model_names), figsize=(17,4.5))
  colors = ['pink', 'lightblue', 'lightgreen', 'orange']
  models_str = ''
  for k in range(len(args.model_names)):
    model_name = args.model_names[k]
    models_str = models_str + model_name + '_'
    # Get models
    if (model_name == "ssd"):
      model = convert("./models/ssd_prunned/deployment/model.onnx")
      img_size = 300
    elif (model_name == "resnet"):
      model = convert("./models/resnet_pruned_nonquant/deployment/model.onnx")
      target_module_list = [nn.Linear]
    elif (model_name == "mobilenetv1"):
      model = convert("./models/mobilenetv1_prunned/deployment/model.onnx")
      target_module_list = [torch.nn.Conv2d]
    elif (model_name == "vgg"):
      model = convert("./models/vgg16_prunned_c/deployment/model.onnx")
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
    model, intern_hooks = Sparse_Pattern_Analyzer.insert_hook(model, target_module_list, args.w_sparsity)
    input_sparsities = [0, 1]
    means_sparse = [[] for i in range(len(intern_hooks))] 
    stddev_sparse = [[] for i in range(len(intern_hooks))] 

    # Run different dataset and record sparsity
    with torch.no_grad():
      layer_random_sparsities = []
      layer_channel_sparsities = []
      network_random_valid_macs = []
      network_channel_valid_macs = []
      mac_propotions = []
      for dataset_path in dataset_paths:
        print ("Processing: ", dataset_path)
        layer_random_sparsity = [[] for i in range(len(intern_hooks))]
        layer_channel_sparsity = [[] for i in range(len(intern_hooks))]
        network_random_valid_mac = []
        network_channel_valid_mac = []
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
            # print ("Shape of output:", output[0].shape, output[1].shape)
            sum_random_sparsity = 0.0 # Calculate network sparsity per sample
            sum_random_v_mac = 0
            sum_channel_v_mac = 0
            sum_t_mac = 0
            random_v_mac = []
            channel_v_mac = []
            for i, hook in enumerate(intern_hooks):
              # print ("hook", hook.sparsity)
              # if (i != 0): break
              if (isinstance(hook.random_sparsity, float)): continue # Layer not running or no sparsity
              random_sparsity = (hook.random_sparsity.cpu())
              random_v_mac.append(hook.random_v_mac.cpu())
              sum_random_v_mac += hook.random_v_mac.cpu()
              layer_random_sparsity[i].append(random_sparsity)

              channel_sparsity = (hook.channel_sparsity.cpu())
              channel_v_mac.append(hook.channel_v_mac.cpu())
              sum_channel_v_mac += hook.channel_v_mac.cpu()
              layer_channel_sparsity[i].append(channel_sparsity)

              sum_t_mac += hook.t_mac

            network_random_valid_mac.append((sum_random_v_mac/sum_t_mac).numpy())
            network_channel_valid_mac.append((sum_channel_v_mac/sum_t_mac).numpy())
            
          # Remove empty list
          layer_random_sparsity = [s for s in layer_random_sparsity if s!=[]]
          layer_random_sparsities.append(layer_random_sparsity)
          network_random_valid_macs.append(network_random_valid_mac)
  
          layer_channel_sparsity = [s for s in layer_channel_sparsity if s!=[]]
          layer_channel_sparsities.append(layer_channel_sparsity)
          network_channel_valid_macs.append(network_channel_valid_mac)
    
    # num_hook_layer = len(layer_sparsities[0])
    # interdata_layer_sparsities = [[] for i in range(num_hook_layer)]
    # for i in range(num_hook_layer):
    #   for j in range(len(dataset_paths)):
    #     interdata_layer_sparsities[i] += layer_sparsities[j][i]
    
    # layer_index = list(range(1, 7)) # Start from begin

    #  Draw layer sparsity
    # if (len(args.model_names) > 1): ax = axs[k]
    # else: ax = axs
    # ax.boxplot([interdata_layer_sparsities[l] for l in layer_index], whis=[0,100], positions=layer_index, patch_artist=True,
    #                 boxprops=dict(facecolor=colors[k],color=colors[k]), medianprops=dict(color='black'))
    # ax.set_ylabel('Sparsity', fontsize=16)
    # ax.set_xlabel('Layer Index', fontsize=16)
    # ax.tick_params(axis='both', labelsize=16)

    colors = ['gold', 'royalblue']
    # Calculate the percentage relative range
    num_hook_layer = len(layer_random_sparsities[0])
    cross_layer_random_valid = [[] for i in range(num_hook_layer)]
    cross_layer_channel_valid = [[] for i in range(num_hook_layer)]
    for i in range(num_hook_layer):
      for j in range(len(dataset_paths)):
        layer_random_valid = [1-x for x in layer_random_sparsities[j][i]]
        layer_channel_valid = [1-x for x in layer_channel_sparsities[j][i]]
        cross_layer_random_valid[i] += layer_random_valid
        cross_layer_channel_valid[i] += layer_channel_valid

      # Normalization by mean
      norm_mean = np.mean([cross_layer_random_valid[i], cross_layer_channel_valid[i]])
      cross_layer_random_valid[i] = [x/norm_mean for x in cross_layer_random_valid[i]]
      cross_layer_channel_valid[i] = [x/norm_mean for x in cross_layer_channel_valid[i]]

      random_mean = np.mean(cross_layer_random_valid[i])
      random_max = np.max(cross_layer_random_valid[i])
      random_min = np.min(cross_layer_random_valid[i])
      print (i, "th layer with random mean ", random_mean,  " with min:", random_min, " and max:", random_max)

      channel_mean = np.mean(cross_layer_channel_valid[i])
      channel_max = np.max(cross_layer_channel_valid[i])
      channel_min = np.min(cross_layer_channel_valid[i])
      print (i, "th layer with channel mean ", channel_mean,  " with min:", channel_min, " and max:", channel_max)

    if (len(args.model_names) > 1): ax = axs[k]
    else: ax = axs
    ax.hist(cross_layer_random_valid[-1], density=True, color=colors[0], bins=50)  # density=False would make counts
    ax.hist(cross_layer_channel_valid[-1], density=True, color=colors[1], bins=50)  # density=False would make counts
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_xlabel('Layer_Valid_Ratio_'+model_name, fontsize=16)
  labels = ['random_sparse', 'channel_sparse']
  fig.legend(labels, ncol=len(labels), bbox_transform=fig.transFigure, loc='upper center', fontsize=16)
  fig.savefig(args.figs_path + models_str + "pattern_sparsity_analysis.pdf")
  plt.close()

    # cross_network_random_valid_macs = []
    # cross_network_channel_valid_macs = []
    # for i in range(len(dataset_paths)):
    #   cross_network_random_valid_macs += network_random_valid_macs[i]
    #   cross_network_channel_valid_macs += network_channel_valid_macs[i]
    # random_mean = np.mean(cross_network_random_valid_macs)
    # random_max = np.max(cross_network_random_valid_macs)
    # random_min = np.min(cross_network_random_valid_macs)
    # print ("Mean valid ratio of random sparsification ", random_mean,  " with min:", random_min, " and max:", random_max)

    # channel_mean = np.mean(cross_network_channel_valid_macs)
    # channel_max = np.max(cross_network_channel_valid_macs)
    # channel_min = np.min(cross_network_channel_valid_macs)
    # print ("Mean valide ratio of channel sparsification ", channel_mean,  " with min:", channel_min, " and max:", channel_max)

  # fig.savefig(args.figs_path + models_str + "layer_sparsity.pdf")
  # plt.close()


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
  parser.add_argument("--model_names", '--names-list', nargs='+', default="resnet", type=str, help="The name of model to collect sparsity", choices=["ssd", "resnet", "mobilenetv1", "vgg"])
  parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  parser.add_argument("--figs_path", default="/path/to/your/directory", type=str, help="The path to all saved figures/images")
  parser.add_argument("--num_samples", default=50, type=int, help="The number of samples from the dataset to calculate sparsity")
  parser.add_argument("--w_sparsity", default=0.95, type=float, help="Sparity of weight")
  parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

  args = parser.parse_args()

  collect_sparse(args)