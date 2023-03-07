import sys
import random
import pandas as pd
import numpy as np
# sys.path.append('../')
from bench_sanger_v3 import calc_sanger_latency

PRIORITY_LIST = [1, 3, 9]

def generate_reqst_table(arrival_rate, num_samples, model_list, lat_lut, sampling_method='Poisson'):
  reqst_table = []
  if sampling_method == 'Poisson': 
    # Each trial follow Exponential (Bernoulli) distribtution, the counting results follow Poisson (Binominal)
    reqst_time = 0.0
    num_models = len(model_list)
    num_priority = len(PRIORITY_LIST)
    for i in range(num_samples):
      reqst_time += random.expovariate(arrival_rate)
      model_str = model_list[random.randint(0, num_models-1)] # Sample model, uniform sampling
      target_lat = lat_lut[model_str]['target_lat']
      priority = PRIORITY_LIST[random.randint(0, num_priority-1)] # Sample priority, uniform sampling
      reqst_table.append((reqst_time, target_lat, model_str, priority))
  else:
    raise NotImplementedError('Sampling approach not supoorted for request construction')
  # print (reqst_table)
  return reqst_table

def construct_lat_table(models, csv_lat_files, args):
  lat_lut = {}
  for i, model in enumerate(models):
    csv_lat_file = csv_lat_files[i]
    print ("Reading from ", csv_lat_file)
    metrics = pd.read_csv(csv_lat_file)
    sparsity = metrics['overall-sparsity']
    num_entries = len(sparsity)
    num_layer = np.max(metrics['layer-indx'])+1
    # Get the latency look-up table for each model
    load_balance = metrics['50%-skip']
    batch_latency_dict = {}
    for i in range(num_entries):
      layer_lat = calc_sanger_latency(sparsity[i], load_balance[i], args.seq_len)
      if metrics['batch-indx'][i] not in batch_latency_dict:
        batch_latency_dict[metrics['batch-indx'][i]] = [ None for i in range(num_layer)]
      batch_latency_dict[metrics['batch-indx'][i]][metrics['layer-indx'][i]] = layer_lat
    # Get the target latency for each model, current method is use mean, can be modified to support others
    e2e_latency = []
    for k, v in batch_latency_dict.items(): # Accumulate all latency in each key
        e2e_latency.append(sum(batch_latency_dict[k]))
    target_lat = np.mean(e2e_latency)

    # Insert into lat_lut dictionary 
    lat_lut[model] = {'lat_lut': batch_latency_dict, 'target_lat': target_lat}
  return lat_lut

class Task:
  def __init__(self, reqst_time, target_lat, model_str, priority):
    self.reqst_time = reqst_time
    self.target_time = self.reqst_time + target_lat
    self.isolated_time = target_lat
    self.finish_time = -1 #initialize as -1
    self.model_str = model_str
    self.lat_queue = []
    self.priority = priority #initialize as -1
    self.urgency = -1
    self.prema_last_exe_time = self.reqst_time # For PREMA use
    self.preama_token = -1 # For PREMA use


  def sample_data(self, num_examples):
    # Assume unifrom sampling
    sample_id = random.randint(0, num_examples-1)
    # More sampling can be support here
    return sample_id

  def construct_task(self, lat_table):
    for i in range(len(lat_table)):
      self.lat_queue.append(lat_table[i])

  def exe(self):
    lat = self.lat_queue.pop(0)
    return lat

  def is_finish(self, sys_time):
    if (len(self.lat_queue) == 0):
      self.finish_time = sys_time
      return True
    else:
      return False