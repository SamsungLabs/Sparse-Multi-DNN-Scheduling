import torch
import torch.nn as nn

def cal_sparsity_tensor(t):
  num_pixel = torch.numel(t)
  num_nonzero = torch.count_nonzero(t)
  # print ("num_pixel:", num_pixel)
  # print ("num_nonzero:", num_nonzero)
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
    self.sparsity = cal_sparsity_tensor(outp)
    # print (m, "with sparsity ", self.sparsity)
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