import torch
import torch.nn as nn
import sys
import numpy as np
import logging
from hw_eyerissv2 import EyerissV2

def cal_sparsity_tensor(t):
  num_pixel = torch.numel(t)
  if (t.dtype is torch.float32):
    num_nonzero = torch.count_nonzero(t)
  elif (t.dtype is torch.quint8):
    num_nonzero = np.count_nonzero(torch.int_repr(t).numpy())
  else: raise ValueeError("Data type for counting non-zero not supported")
  sparsity = 1 - num_nonzero/num_pixel
  # print (sparsity)
  return sparsity


class Stat_Collector_Eyerissv2:
  def __init__(self, m):
    self.handle = m.register_forward_hook(self.hook_fn)
    self.sparsity = 0.0
    self.eyerissv2_simulator = EyerissV2()
    self.estimated_cycle = -1
    self.iact_sparsity = -1

  def hook_fn(self, m, inp, outp):
    estimated_cycle = 0 
    self.out_features = outp.clone()
    self.in_features = inp
    self.m = m
    if (self.out_features.dtype is torch.float32):
      weight = self.m.weight
    else:
      weight = self.m.weight()
    w_shape = weight.shape
    o_shape = self.out_features.shape
    device = weight.device
    w_mask = torch.where(weight != 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    i_mask = torch.where(self.in_features[0] != 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    if (isinstance(self.m, nn.ReLU)):
      # ReLU Sparsity
      self.sparsity = cal_sparsity_tensor(outp)
      # print ("ReLU Sparsity:", self.sparsity)
    elif (isinstance(self.m, (nn.Linear, torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU, torch.nn.quantized.modules.linear.Linear))):
      # Get estimated cycles of Linear
      self.estimated_cycle = self.eyerissv2_simulator.simulate(i_mask, w_mask, 'FC')
      self.iact_sparsity = cal_sparsity_tensor(i_mask).item()
      logging.debug("Simulate Linear with input-activation sparisty %f on Eyerissv2. The estimated cycle is %d."%(self.iact_sparsity, self.estimated_cycle))
    elif (isinstance(self.m, (nn.Conv2d, torch.nn.quantized.modules.conv.Conv2d, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d))):
      # Get estimated cycles of Conv
      self.estimated_cycle = self.eyerissv2_simulator.simulate(i_mask, w_mask, 'CONV', self.m.groups)
      self.iact_sparsity = cal_sparsity_tensor(i_mask).item()
      logging.debug("Simulate Conv with input-activation sparisty %f on Eyerissv2. The estimated cycle is %d."%(self.iact_sparsity, self.estimated_cycle))
    else:
      raise NameError("Hook Layer not supported")

  
  def remove(self):
    self.handle.remove()

# Insert hook of every "target_module"
# Return the inserted model and intermediate result 
def insert_hook_simulator(model, target_module_list):
  intern_outputs = []
  for layer in model.modules():
    for target_module in target_module_list:
      if isinstance(layer, target_module):
        intern_outputs.append(Stat_Collector_Eyerissv2(layer))
        print ("Insert hook into ", layer.__class__)

  return model, intern_outputs