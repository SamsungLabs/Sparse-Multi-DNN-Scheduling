import math
import torch
import torch.nn as nn
import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'dataset_sparsity')))
from data_util import create_sparse_tensor 
import logging

# Spec of EyerissV2 according to the Table II of EyerissV2 paper (https://arxiv.org/pdf/1807.07928.pdf)
EyerissV2_Configs = {
  'IACT_SPAD' : {
    'Height' : 16,
    'Width' : 12,
  },

  'IACT_GLB' : {
    'Number' : 3,
    'Size' : 4 * 1000 * 8, # original is 1.5 KB, we have to increase this to 2.5 KB to run SSD
  },

  'Cluster_Array' : {
    'cluster_row' : 8,
    'cluster_clomn' : 2,
  },

  'PE_Cluster' : {
    'pe_row' : 3,
    'pe_clomn' : 4,
  },

  'Router_Cluster' : {
    'IAct_Router': {},
    'Weight_Router': {},
    'PSum_Router': {},
  },

  'GLB_Cluster' : {
    'IAct_SRAM': {},
    'PSUM_SRAM': {},
  },

  'External_DDR' : {
    'Bandwidth' : 25600, # MB/s
  },
}


# Class Eyeriss Hardware

class EyerissV2:
  def __init__(self, Clock_Freq = 200, Num_PE_Row = 3, Num_PE_Column = 4, Num_Cluster_Row = 8, Num_Cluster_Column = 2,
                Weight_BitWidth = 8, IAct_BitWidth = 8):
    '''
    For the notation, we follow the original Eyeriss-V2 Paper (https://arxiv.org/pdf/1807.07928.pdf)
    '''
    self.Clock_Freq = Clock_Freq # In MHz
    self.Weight_BitWidth = Weight_BitWidth
    self.IAct_BitWidth = IAct_BitWidth
    self.Parallel_Read = True
    self.PE_Num = Num_PE_Row * Num_PE_Column * Num_Cluster_Row * Num_Cluster_Column
    self.layer_type = None
    self.group = -1
    # Single PE Level
    # IAct parallism of single PE
    self.C0 = -1 # Input Feature Channel, parallelsim inside PE, R*C0 < IACT_SPAD (16X12b)
    self.R = -1 # Kernel Height
    self.E = -1 # Output Feature Height
    self.F0 = -1 # Output Feature Width, parallelsim inside PE, E*F0*R*C0 < IACT_GLB (Three GLB, each has 1.5 kB)
    # Weight parallism of single PE
    self.M0 = -1 # Output Feature Channel (filter), parallelsim inside PE, M0*R*C0 < WEIGHT_SPAD (96 x 24b)
    self.Num_PE_Row = Num_PE_Row
    self.Num_PE_Column = Num_PE_Column
    self.Num_Cluster_Row = Num_Cluster_Row
    self.Num_Cluster_Column = Num_Cluster_Column

    # Single Cluster Level (a group of PEs)
    self.S1 = Num_PE_Row # Kernel Width, parallelsim inside cluster
    self.F1 = Num_PE_Column # Output Feature Width, parallelsim inside cluster

    # Mesh Network Mapping (a group of Clusters)
    self.C1 = Num_Cluster_Row
    self.M1 = Num_Cluster_Column

    # Temporal Mapping
    self.M2 = -1 # Output Feature Channel (filter), temporal mapping, M0*M1*M2 = M
    self.F2 = -1 # Output Feature Width, temporal mapping, F0*F1*F2 = F
    self.C2 = -1 # Input Feature Channel, temporal mapping, C0*C1*C2 = M
    self.S2 = -1 # Kernel Width, temporal mapping, S1*S2 = S

  def mapping(self, iact_shape, weight_shape):
    if (self.layer_type == 'CONV'):
      out_channel, _, kernel_height, kernel_width = weight_shape
      out_channel = out_channel/self.group # Support for DW conv
      _, in_channel, in_height, in_width = iact_shape
    elif (self.layer_type == 'FC'):
      out_channel, in_channel = weight_shape
      in_height, in_width, kernel_height, kernel_width = 1, 1, 1, 1
    else:
      raise RuntimeError('Layer not supported in current simulator')

    out_width, out_height = in_width, in_height

    # Determine the mapping configuration
    # Mapping of single PE

    self.R = kernel_height
    self.C0 = EyerissV2_Configs['IACT_SPAD']['Height'] // self.R # Input Feature Channel, parallelsim inside PE, R*C0 < IACT_SPAD (16X12b)
    if (in_channel <=3):
      self.C0 = in_channel # Usually the first layer
    else:
      self.C0 = 2**int(math.log(self.C0, 2)) # largest power of 2 less then C0
      self.C0 = self.C0 if (self.C0 <= in_channel) else in_channel

    self.E = out_width
    IACT_GLB_SIZE = EyerissV2_Configs['IACT_GLB']['Number'] * EyerissV2_Configs['IACT_GLB']['Size']
    self.F0 = IACT_GLB_SIZE // (self.E * self.R * self.C0 * self.IAct_BitWidth) # Output Feature Width, parallelsim inside PE, E*F0*R*C0 < IACT_GLB (Three GLB, each has 1.5 kB)

    self.F0 = 2**int(math.log(self.F0, 2)) # largest power of 2 less then F0
    self.F0 = self.F0 if (self.F0 <= out_width) else out_width

    # In the ideal case, M0 should be set according to the weight sparsity and buffer size of weight SPad. 
    # We follow the original paper to set it as 32, which is also the comon case shown in their Table-III
    self.M0 = 32 if out_channel > 32 else out_channel

    # Mapping of cluster is fixed and determined before running
    self.C1 = self.Num_Cluster_Row
    self.C1 = self.C1 if (self.C1 * self.C0 <= in_channel) else int(in_channel/self.C0)
    self.F1 = self.Num_PE_Column
    self.F1 = self.F1 if (self.F1 * self.F0 <= out_width) else int(out_width/self.F0)
    self.S1 = self.Num_PE_Row if kernel_width >= self.Num_PE_Row else kernel_width
    self.M1 = self.Num_Cluster_Column
    self.M1 = self.M1 if (self.M1 * self.M0 <= out_channel) else int(out_channel/self.M0)

    # Mapping of mesh network
    self.M2 = int(math.ceil(out_channel / (self.M0 * self.M1))) # Output Feature Channel (filter), temporal mapping, M0*M1*M2 = M
    self.F2 = int(math.ceil(out_width / (self.F0 * self.F1))) # Output Feature Width, temporal mapping, F0*F1*F2 = F
    self.C2 = int(math.ceil(in_channel / (self.C0 * self.C1))) # Input Feature Channel, temporal mapping, C0*C1*C2 = M
    self.S2 = int(math.ceil(kernel_width / self.S1)) # Kernel Width, temporal mapping, S1*S2 = S

    logging.debug("in_width:%d, out_width:%d, M2:%d, F2:%d, C2:%d, S2:%d, M1:%d, F1:%d, C1:%d, S1:%d, C0:%d, R:%d, E:%d, F0:%d, M0:%d" % (in_width, out_width, 
                  self.M2, self.F2, self.C2, self.S2, self.M1, self.F1, self.C1, self.S1, self.C0, self.R, self.E, self.F0, self.M0))

  def run(self, IAct_tensor, Weight_tensor):

    estimated_cycle = 0
    parallel_cycle = 0
    # For valid oepration, spend four cycle to load weight, compute and write back
    #   corresponding to https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L293
    #   and https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L296
    if self.layer_type == 'CONV':
      valid_mac = torch.nn.functional.conv2d(IAct_tensor, Weight_tensor, padding='same', stride=1, groups=self.group).sum()
    elif self.layer_type == 'FC':
      # FC is equivalent to CONV 1X1 with input height and weight are both equal to 1
      valid_mac = torch.nn.functional.linear(IAct_tensor, Weight_tensor).sum() # (N, C)
      IAct_tensor = torch.unsqueeze(IAct_tensor, -1) # (N, C, H)
      IAct_tensor = torch.unsqueeze(IAct_tensor, -1) # (N, C, H, W)
    else:
      raise RuntimeError('Layer not supported in current simulator while running')
    parallel_cycle += 4 * valid_mac
    
    logging.debug("After counting valid operation, parallel cycle:%d"%(parallel_cycle))

    # Load IAct and Weight from main/external memory to GLB
    # We assume double buffering and the loading time is overlapped with computation,
    #      so this cost will be only introducecd in the first round of Mesh Network mapping (M1/C1/S1/F1), 
    #      and thus not related to temporal mapping (M2/C2/S2/F2)
    # The cost asscosiated with https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L200

    # Load data from main to glb

    # Load weight from main to glb

    # Get the number of non-zeros in each [Noc][GLB] pad

    # Assuming the input tensor is (N, C, H, W)
    IAct_shape = IAct_tensor.shape
    num_batch, num_channel, height, width = IAct_shape
    assert num_batch == 1
    assert num_channel == self.C2 * self.C1 * self.C0
    if (width != self.F2 * self.F1 * self.F0):
      logging.debug("Padding the width from %d to %d" % (width, self.F2 * self.F1 * self.F0))
      pad_width = self.F2 * self.F1 * self.F0 - width
      pad_height = pad_width
      IAct_tensor = torch.nn.functional.pad(IAct_tensor, (0, pad_height, 0, pad_width), "constant", 0) 
      width = self.F2 * self.F1 * self.F0
      height = width
    IAct_tensor.permute(0, 1, 3, 2) # (N, C, W, H)
    IAct_tensor = torch.reshape(IAct_tensor, (self.C2, self.C1, self.C0, self.F2, self.F1, self.F0, height)) # (C2, C1, C0, F2, F1, F0, H)
    IAct_tensor.permute(0, 1, 3, 4, 5, 6, 2) # (C2, C1, F2, F1, F0, H, C0)
    nonzero_C0 = torch.sum(IAct_tensor, -1) # (C2, C1, F2, F1, F0, H), count the number of zeros per C0

    # Since each row is R * C0, accumulate each R rows in sliding window manner
    shift_nonzero_C0 = nonzero_C0.clone().detach()
    nonzero_R_C0 = nonzero_C0
    for _ in range(self.R - 1):
      shift_nonzero_C0 = torch.roll(shift_nonzero_C0, -1, -1) 
      shift_nonzero_C0[:, :, :, :, :, -1] = 0
      nonzero_R_C0 = nonzero_R_C0 + shift_nonzero_C0

    nonzero_R_C0 = nonzero_R_C0.view(*nonzero_C0.shape[:4], -1) # (C2, C1, F2, F1, F0 * H), count the number of zeros per R * C0  torch.count_nonzero(x, dim=0)

    nonzero_column_IActNocSPad = torch.count_nonzero(nonzero_R_C0, dim=-1) # (C2, C1, F2, F1), count the number of non-zeros rows in each (F0 * H) *  (R * C0) IActNocSPad
    
    # Read from GBL to scratch pad (input activation). Even there are all zeros, it still takes one cycle to check
    #   corresonding to https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L271
    act_check_cycle = self.M1 * self.F1 * self.C1 * self.S1 * self.M2 * self.F2 * self.C2 * self.S2 * self.E * self.F0
    parallel_cycle += act_check_cycle

    logging.debug("act address read, parallel cycle:%d" % (act_check_cycle))
    # For every column with non-zeros, spend two more cycles to 1. read data; 2. load weight address to check
    #   corresponding to https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L279
    #   and https://github.com/SingularityKChen/dl_accelerator/blob/876a5eb57d906119f4f0d85f3ae28ace1f444923/dla/tests/src/ScalaModelTest.scala#L285
    weight_check_cycle =  2 * (self.M1 * self.S1 * self.M2 * self.S2) * torch.sum(nonzero_column_IActNocSPad).item()
    parallel_cycle += weight_check_cycle

    logging.debug("weight address read, parallel cycle:%d" % (weight_check_cycle))

    estimated_cycle += int(parallel_cycle/self.PE_Num) # We assume all the PEs can be used.
    return estimated_cycle



  def simulate(self, i_mask, w_mask, layer_type = 'CONV', group = 1):
    iact_shape = i_mask.shape
    weight_shape = w_mask.shape
    self.layer_type = layer_type
    self.group = group
    # Set the mapping information according to the shape of current layer
    self.mapping(iact_shape, weight_shape)

    estimated_cycle = self.run(i_mask, w_mask)

    return estimated_cycle

if __name__ == '__main__':
  '''
  Unit test of Eyeriss simulator, a simple convolution
  '''
  N = 1 # batch
  C = 16 # input channel
  H = 56 # infeature height
  W = 56 # infeature width
  F = 16 # output channel
  R = 3 # kernel height
  S = 3 # kernel width
  G = [1, C]
  clock_freq = 200 # In MHz
  sparsities_inpt = [0.2, 0.7]
  sparsities_weight = [0.2, 0.7]
  for g in G:
    for i in range(len(sparsities_inpt)):
      print ("-"*100)
      sparsity_inpt = sparsities_inpt[i]
      sparsity_weight = sparsities_weight[i]
      print ("input sparsity:", sparsity_inpt) 
      print ("weight sparsity:", sparsity_weight)
      inpt_shape = (N, C, H, W)
      test_inpt = create_sparse_tensor(inpt_shape, sparsity_inpt)
      
      test_m = nn.Conv2d(C, F, R, stride=1, groups=g)
      # Get mask
      weight = test_m.weight
      device = weight.device
      weight_shape = weight.shape
      weight = create_sparse_tensor(weight_shape, sparsity_weight)
      test_m.weight = nn.Parameter(weight)

      w_mask = torch.where(weight != 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
      i_mask = torch.where(test_inpt != 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

      test_outpt = test_m(test_inpt)

      print ("weight:", w_mask.shape)
      print ("input:", i_mask.shape)
      print ("output:", test_outpt.shape) 

      test_hw = EyerissV2()
      estimated_cycle = test_hw.simulate(i_mask, w_mask, group=g)
      print ("Standard Conv with group %f, estimated cycle:%f" % (g, estimated_cycle / (clock_freq*1000000)))