import torch
import torch.nn as nn
import sys
import numpy as np

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

def create_sparse_tensor(shape, p):
  dense_tensor = torch.randn(shape)
  sparse_tensor = torch.where(torch.rand(shape) < p, torch.tensor(0.), dense_tensor)
  return sparse_tensor

class Stat_Collector:
  def __init__(self, m):
    self.handle = m.register_forward_hook(self.hook_fn)
    self.sparsity = 0.0
  def hook_fn(self, m, inp, outp):
    self.out_features = outp.clone()
    self.in_features = inp
    self.m = m
    if (isinstance(self.m, nn.ReLU) or isinstance(self.m, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d)):
      # ReLU Sparsity
      self.sparsity = cal_sparsity_tensor(outp)
      # print ("ReLU Sparsity:", self.sparsity)
    elif (isinstance(self.m, nn.Conv2d)):
      w_shape = self.m.weight.shape
      o_shape = self.out_features.shape
      self.t_mac = w_shape.numel() * o_shape[-1] * o_shape[-1] * o_shape[0] # total mac counts Filter * Channel * Kernel_size * Kernel_size * Output_size * Output_size * Bath_size
      w_mask = torch.where(self.m.weight != 0, torch.tensor(1.0), torch.tensor(0.0))
      i_mask = torch.where(self.in_features[0] != 0, torch.tensor(1.0), torch.tensor(0.0))
      self.v_mac = torch.nn.functional.conv2d(i_mask, w_mask, padding=self.m.padding, stride=self.m.stride).sum()
      # Conv Sparsity
      self.sparsity = 1- self.v_mac/self.t_mac
      # print ("total mac:", self.t_mac, " valid mac:", self.v_mac)
      # print ("Conv Sparsity:", self.sparsity)
      if (self.sparsity > 1  or self.sparsity <0):
        print ("w_shape:", w_shape)
        print ("o_shape:", o_shape)
        print ("total mac:", self.t_mac, " valid mac:", self.v_mac)
        print ("input mask", i_mask.shape)
        print ("weight mask", w_mask.shape)
        sys.exit(0)
    else:
      raise NameError("Hook Layer not supported")
  def remove(self):
    self.handle.remove()

  # Insert hook of every "target_module"
  # Return the inserted model and intermediate result 
  def insert_hook(model, target_module_list):
    intern_outputs = []
    for layer in model.modules():
      for target_module in target_module_list:
        if isinstance(layer, target_module):
          # print("Collect: %s" % (layer))
          intern_outputs.append(Stat_Collector(layer))
    return model, intern_outputs