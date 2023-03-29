'''
Use SparseML to get sparse CNN models with different sparsity patterns.
Mainly adopted from here: https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/classification.ipynb
'''
# Externel
import sparseml
import sparsezoo
import torch
import torchvision

from sparseml.pytorch.models import ModelRegistry

from sparseml.pytorch.optim import (
    ScheduledModifierManager,
)

from tqdm.auto import tqdm
import math

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity

import sys
import argparse
import logging
import random

# Internel
from train_utils import run_classification_one_epoch, get_datasets

Pretrainted_Model_Dataset = {"resnet50": "imagenette",
                            "vgg16": "imagenet",
                            "mobilenet": "imagenette",
                            "efficientnet": "imagenet"}

def prune_model(args):
  #######################################################
  # Define your model below
  #######################################################
  print("constructing model...")
  model = ModelRegistry.create(
      key=args.model,
      pretrained=True,
      pretrained_dataset=Pretrainted_Model_Dataset[args.model],
      num_classes=args.num_class,
  )
  input_shape = ModelRegistry.input_shape(args.model)
  input_size = input_shape[-1]
  print(model)
  print('input size:', input_size)

  #######################################################
  # Define your train and validation datasets below
  #######################################################

  # print("\nloading train dataset...")
  # if (args.dataset == "imagenette"):
  #   train_dataset = ImagenetteDataset(
  #       train=True, dataset_size=ImagenetteSize.s320, image_size=input_size
  #   )
  # else:
  #   raise NameError('Dataset not supported')
  # print(train_dataset)

  # print("\nloading val dataset...")
  # if (args.dataset == "imagenette"):
  #   val_dataset = ImagenetteDataset(
  #       train=False, dataset_size=ImagenetteSize.s320, image_size=input_size
  #   )
  # else:
  #   raise NameError('Dataset not supported')
  # print(val_dataset)

  train_dataset, val_dataset = get_datasets(args, input_size)

  # setup device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  print("Using device: {}".format(device))

  # setup data loaders
  batch_size = args.batch_size
  train_loader = DataLoader(
      train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=8
  )
  val_loader = DataLoader(
      val_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=8
  )

  # setup loss function and optimizer, LR will be overriden by sparseml
  criterion = CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)



  from sparsezoo import Model, search_models
  if args.recipe_path is None:
    zoo_model = search_models(
        domain="cv",
        sub_domain="classification",
        architecture="mobilenet_v1",
        sub_architecture="1.0",
        framework="pytorch",
        repo="sparseml",
        dataset="imagenet",
        sparse_name="pruned",
    )[0]  # unwrap search result
    recipe_path = zoo_model.recipes.default.path
    print(f"Recipe downloaded to: {recipe_path}")
  else:
    recipe_path = args.recipe_path

  save_model_path = args.save_model_path + args.recipe_path.split('/')[-1].split('.')[0] + '.pt'
  print("\nmodel will be saved to %s"%(save_model_path))
  # create ScheduledModifierManager and Optimizer wrapper
  manager = ScheduledModifierManager.from_yaml(recipe_path)
  optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))



  # Run model pruning
  epoch = manager.min_epochs
  while epoch < manager.max_epochs:
      # run training loop
      epoch_name = "{}/{}".format(epoch + 1, manager.max_epochs)
      print("Running Training Epoch {}".format(epoch_name))
      train_loss, train_acc = run_classification_one_epoch(
          model, train_loader, criterion, device, train=True, optimizer=optimizer
      )
      print(
          "Training Epoch: {}\nTraining Loss: {}\nTop 1 Acc: {}\n".format(
              epoch_name, train_loss, train_acc
          )
      )
      
      # run validation loop
      print("Running Validation Epoch {}".format(epoch_name))
      val_loss, val_acc = run_classification_one_epoch(
          model, val_loader, criterion, device
      )
      print(
          "Validation Epoch: {}\nVal Loss: {}\nTop 1 Acc: {}\n".format(
              epoch_name, val_loss, val_acc
          )
      )
      
      epoch += 1

  manager.finalize(model)

  # print sparsities of each layer
  for (name, layer) in get_prunable_layers(model):
      print("{}.weight: {:.4f}".format(name, tensor_sparsity(layer.weight).item()))


  # Save models

  torch.save(model, save_model_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  parser.add_argument("--model", default="resnet50", type=str, choices=["resnet50", "vgg16", "mobilenet", "efficientnet"],
                      help="The name of candidate model to prune.")
  parser.add_argument("--dataset", default="imagenette", type=str, choices=["imagenette"],
                      help="The name of dataset to pre-train.")
  parser.add_argument("--recipe_path", default=None, type=str, help="The path to pruning receipt")
  parser.add_argument("--save_model_path", default="./recipes_models/", type=str, help="The path to save models")
  parser.add_argument("--batch_size", default=32, type=int, required=False,
                      help="The batch size for training and inference.")
  parser.add_argument("--num_class", default=10, type=int, required=False,
                      help="The number of classes of dataset.")
  # Verbosity / Logging
  parser.add_argument("--debug", action="store_true")

  # Random seed
  parser.add_argument("--seed", type=int, default=1,
                      help="Random seed.")

  # parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  # parser.add_argument("--w_sparsity", default=0.95, type=float, help="Sparity of weight")

  args = parser.parse_args()


  # Logging setup
  if args.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  random.seed(args.seed)

  prune_model(args)
