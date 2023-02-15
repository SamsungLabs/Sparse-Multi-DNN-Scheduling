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
      # Linear Sparsity
      self.t_mac = w_shape.numel() * o_shape[-2] # total mac counts In_Channel * Out_Channel * Batch_size
      self.v_mac = torch.nn.functional.linear(i_mask, w_mask).sum()
      self.sparsity = 1- self.v_mac/self.t_mac
      # print ("Linear Sparsity:", self.sparsity)
    elif (isinstance(self.m, (nn.Conv2d, torch.nn.quantized.modules.conv.Conv2d, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d))):
      # Conv Sparsity
      self.t_mac = w_shape.numel() * o_shape[-1] * o_shape[-1] * o_shape[0] # total mac counts Filter * Channel * Kernel_size * Kernel_size * Output_size * Output_size * Bath_size
      self.v_mac = torch.nn.functional.conv2d(i_mask, w_mask, padding=self.m.padding, stride=self.m.stride, groups=self.m.groups).sum()
      self.sparsity = 1- self.v_mac/self.t_mac
      # print ("total mac:", self.t_mac, " valid mac:", self.v_mac)
      # print ("Conv Sparsity:", self.sparsity)
    else:
      raise NameError("Hook Layer not supported")
    if (self.sparsity > 1  or self.sparsity <0):
      print (self.m)
      print ("w_shape:", w_shape)
      print ("o_shape:", o_shape)
      print ("total mac:", self.t_mac, " valid mac:", self.v_mac)
      print ("input mask", i_mask.shape)
      print ("weight mask", w_mask.shape)
      raise ValueError("The sparsity should between 0 and 1")
  def remove(self):
    self.handle.remove()

  # Insert hook of every "target_module"
  # Return the inserted model and intermediate result 
  def insert_hook(model, target_module_list):
    intern_outputs = []
    for layer in model.modules():
      # print (layer.__class__)
      for target_module in target_module_list:
        if isinstance(layer, target_module):
          # print("Collect: %s" % (layer))
          intern_outputs.append(Stat_Collector(layer))
    return model, intern_outputs



class Sparse_Pattern_Analyzer:
  def __init__(self, m, w_sparsity):
    self.handle = m.register_forward_hook(self.hook_fn)
    self.random_sparsity = 0.0
    self.channel_sparsity = 0.0
  def hook_fn(self, m, inp, outp):
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
    # print ("Test1")
    self.w_valid_ratio = w_mask.sum()/w_shape.numel()
    if (self.w_valid_ratio == 1.0): 
      # print ("No Sparsity Found in", self.m)
      return
    # print ("w_mask:", w_mask.sum())
    # print ("w_element:", w_shape.numel())
    if (isinstance(self.m, (nn.Linear, torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU, torch.nn.quantized.modules.linear.Linear))):
      # Linear Sparsity
      num_valid_ifeature = int(w_shape[-1]*self.w_valid_ratio)
      # Calculate the sparsity created by random sparsification
      self.t_mac = w_shape.numel() * o_shape[-2] # total mac counts In_Channel * Out_Channel * Batch_size
      self.random_v_mac = torch.nn.functional.linear(i_mask, w_mask).sum()
      self.random_sparsity = 1- self.random_v_mac/self.t_mac
      # print ("Valide random mac op:", self.random_v_mac)

      # Calculate the sparsity created by random sparsification
      valid_i_mask = i_mask[:num_valid_ifeature] # Channel prunning, assume only the first few channels are valid
      valid_i_ratio = valid_i_mask.sum() / valid_i_mask.shape.numel()
      self.channel_v_mac = self.t_mac * self.w_valid_ratio * valid_i_ratio # Channel sparsity and input sparsity
      self.channel_sparsity = 1 - self.channel_v_mac/self.t_mac
      # print ("Valide channel:", num_valid_ifeature, "/", w_shape[-1], " valid ratio:", self.w_valid_ratio)
      # print ("Valide random mac op:", self.random_v_mac, "  Vs.   Valide channel mac op:", self.channel_v_mac)
      # print ("Valide random valid ratio:", self.random_v_mac/self.t_mac, "  Vs.   Valide channel valid ratio:", self.channel_v_mac/self.t_mac)
    elif (isinstance(self.m, torch.nn.Conv2d)):
      # Conv Sparsity
      num_valid_ifeature = int(w_shape[-3]*self.w_valid_ratio) # Filter * Channel * Kernel_size * Kernel_size, so -3 means Channel
      self.t_mac = w_shape.numel() * o_shape[-1] * o_shape[-1] * o_shape[0] # total mac counts Filter * Channel * Kernel_size * Kernel_size * Output_size * Output_size * Bath_size
      self.random_v_mac = torch.nn.functional.conv2d(i_mask, w_mask, padding=self.m.padding, stride=self.m.stride, groups=self.m.groups).sum()
      self.random_sparsity = 1- self.random_v_mac/self.t_mac

      valid_i_mask = i_mask[:num_valid_ifeature] # Channel prunning, assume only the first few channels are valid
      valid_i_ratio = valid_i_mask.sum() / valid_i_mask.shape.numel()
      self.channel_v_mac = self.t_mac * self.w_valid_ratio * valid_i_ratio # Channel sparsity and input sparsity
      # print ("valid_i_mask.sum():", valid_i_mask.sum())
      # print ("valid_i_mask.shape.numel()", valid_i_mask.shape.numel())
      self.channel_sparsity = 1 - self.channel_v_mac/self.t_mac
    else:
      raise NameError("Hook Layer not supported")
  def remove(self):
    self.handle.remove()

  # Insert hook of every "target_module"
  # Return the inserted model and intermediate result 
  def insert_hook(model, target_module_list, w_sparsity):
    intern_outputs = []
    for layer in model.modules():
      # print (layer.__class__)
      for target_module in target_module_list:
        if isinstance(layer, target_module):
          if isinstance(layer, torch.nn.Conv2d):
            if (layer.kernel_size[0]>1): continue
          print("Collect: %s" % (layer))
          intern_outputs.append(Sparse_Pattern_Analyzer(layer, w_sparsity))
    return model, intern_outputs