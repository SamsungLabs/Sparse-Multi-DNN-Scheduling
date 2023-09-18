import sys
import random
import pandas as pd
import numpy as np
import copy

from sklearn import linear_model

#########################################
############## Latency Pred ##############
#########################################

def avg_pred_linear_rate(measured_sparsities, avg_sparsities):
  """
  This estimator uses the average of the ratios between the real and average output sparsity 
  of all already executed layers in the task to predict the sparse latency of the current layer.

  Args:
    measured_sparsities: Actual sparsity of each already executed layer in the task.
    avg_spartsities: Average sparsity of each layer in the task across the target dataset. 
  """
  num_exe_layer = len(measured_sparsities)
  linear_rate = 1.0
  if (num_exe_layer > 0):
    sparsity_ratios = []
    for i in range(num_exe_layer):
      sparsity_ratio = (1 - measured_sparsities[i]) / (1 - avg_sparsities[i])
      sparsity_ratios.append(sparsity_ratio)
    linear_rate = sum(sparsity_ratios) / num_exe_layer # Average the ratio to get linear rate
    # print ("avg value:",  linear_rate, " sparsity ratio list:", sparsity_ratios)
    # print ("sparsity measured list:",  measured_sparsities, " sparsity real list:", avg_sparsities)
  return linear_rate


def last_one_pred_linear_rate(measured_sparsities, avg_sparsities):
  """
  This estimator uses the ratio between the real and average output sparsity 
  of the previous layer in the task to predict the sparse latency of the current layer.
  
  Args:
    measured_sparsities: Actual sparsity of each already executed layer in the task.
    avg_spartsities: Average sparsity of each layer in the task across the target dataset. 
  """
  num_exe_layer = len(measured_sparsities)
  linear_rate = 1.0
  if (num_exe_layer > 0):
    last_indx = num_exe_layer-1
    linear_rate = (1 - measured_sparsities[last_indx]) / (1 - avg_sparsities[last_indx])
    # print ("last one value:",  linear_rate)
    # print ("sparsity measured list:",  measured_sparsities, " sparsity real list:", avg_sparsities)
  return linear_rate

def last_N_pred_linear_rate(measured_sparsities, avg_sparsities, window_size=3):
  """
  This estimator uses the average of the ratios between the real and average output sparsity 
  of the last N already executed layers in the task to predict the sparse latency of the current layer.

  Args:
    measured_sparsities: Actual sparsity of each already executed layer in the task.
    avg_spartsities: Average sparsity of each layer in the task across the target dataset. 
  """
  num_exe_layer = len(measured_sparsities)
  linear_rate = 1.0
  if (num_exe_layer > 0):
    sparsity_ratios = []
    start_indx = num_exe_layer - window_size
    start_indx = 0 if start_indx < 0 else start_indx # Boudary check
    for i in range(start_indx, num_exe_layer):
      sparsity_ratio = (1 - measured_sparsities[i]) / (1 - avg_sparsities[i])
      sparsity_ratios.append(sparsity_ratio)
    linear_rate = sum(sparsity_ratios) / window_size # Average the ratio for last N to linear rate
    # print ("last N value:",  linear_rate, " sparsity ratio list:", sparsity_ratios)
    # print ("sparsity measured list:",  measured_sparsities, " sparsity real list:", avg_sparsities)
  return linear_rate


def regression_train(train_dataset):
  """
  Train a linear regressor to estimate the network (end-to-end) based on the sparsity of current layer 

  Args:
    measured_sparsities: Actual sparsity of each already executed layer in the task.
    avg_spartsities: Average sparsity of each layer in the task across the target dataset. 
  """
  model_coefs = {}
  for model, v_dict in train_dataset.items():
    lat_lut = v_dict['lat_lut']
    avg_lat = v_dict['avg_lat_per_pattern']
    avg_sparsity = v_dict['avg_sparsity']
    sparsity_lut = v_dict['sparsity_lut']
    num_samples = len(lat_lut) 
    num_layer = len(lat_lut[0])
    coefs = []
    for layer_indx in range(1, num_layer):
      reg = linear_model.LinearRegression()
      # Get sparsity
      xs = []
      ys = []
      for sample_indx in range(num_samples):
        real_lat = sum(lat_lut[sample_indx][layer_indx:])
        est_lat = sum(avg_lat[layer_indx:])
        # xs.append([sparsity_lut[sample_indx][layer_indx]])
        # ys.append(real_lat)
        xs.append([(1 - sparsity_lut[sample_indx][layer_indx]) / (1 - avg_sparsity[layer_indx])])
        ys.append(real_lat/est_lat)
      # plt.scatter(xs, ys)
      # plt.savefig("temp" + str(layer_indx) +".pdf", bbox_inches='tight')
      # plt.close()
      reg.fit(xs, ys)
      coefs.append(reg.coef_)
    model_coefs[model] = coefs
    print (model, model_coefs[model])
  return model_coefs

def linear_reg_pred(measured_sparsities, avg_sparsities, coefs):
  num_exe_layer = len(measured_sparsities)
  est_lat = 0.0
  if (num_exe_layer > 0):
    last_indx = num_exe_layer-1
    coef = coefs[last_indx]
    est_lat = coef[0] * (measured_sparsities[last_indx])
  return est_lat

