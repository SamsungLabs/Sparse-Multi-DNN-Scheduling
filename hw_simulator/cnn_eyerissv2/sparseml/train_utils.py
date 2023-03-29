'''
Use SparseML to get sparse CNN models with different sparsity patterns.
Mainly adopted from here: https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/classification.ipynb
'''
from tqdm.auto import tqdm
import math
import torch
from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
from sparseml.pytorch.datasets import VOCDetectionDataset
from sparseml.pytorch.utils import get_default_boxes_300
from sparseml.pytorch.utils import DEFAULT_LOSS_KEY


def run_classification_one_epoch(model, data_loader, criterion, device, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs, _ = model(inputs)  # model returns logits and softmax as a tuple
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_predictions += inputs.size(0)

    loss = running_loss / (step + 1.0)
    accuracy = total_correct / total_predictions
    return loss, accuracy


def run_detection_one_epoch(model, data_loader, criterion, device, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = [
            label.to(device) if isinstance(label, torch.Tensor) else label
            for label in labels
        ]

        if train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion((inputs, labels), outputs)[DEFAULT_LOSS_KEY]

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    loss = running_loss / (step + 1.0)
    return loss



def get_datasets(args, input_size):

  #######################################################
  # Define your train and validation datasets below
  #######################################################

  print("\nloading train dataset...")
  if (args.dataset == "imagenette"):
    train_dataset = ImagenetteDataset(
        train=True, dataset_size=ImagenetteSize.s320, image_size=input_size
    )
  elif(args.dataset == "voc"):
    default_boxes = get_default_boxes_300("voc")
    train_dataset = VOCDetectionDataset(
        train=True, rand_trans=True, preprocessing_type="ssd", default_boxes=default_boxes
    )
  else:
    raise NameError('Dataset not supported')
  print(train_dataset)

  print("\nloading val dataset...")
  if (args.dataset == "imagenette"):
    val_dataset = ImagenetteDataset(
        train=False, dataset_size=ImagenetteSize.s320, image_size=input_size
    )
  elif(args.dataset == "voc"):
    val_dataset = VOCDetectionDataset(
        train=False, preprocessing_type="ssd", default_boxes=default_boxes
    )
  else:
    raise NameError('Dataset not supported')
  print(val_dataset)

  return train_dataset, val_dataset
