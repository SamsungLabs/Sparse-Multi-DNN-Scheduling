import sys
import random
import pandas as pd
import numpy as np
import copy
from bench_sanger_v3 import calc_sanger_latency
import matplotlib.pyplot as plt
import os
from itertools import islice

PRIORITY_LIST = [1, 3, 9] # PREMA's priority scheme

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def generate_reqst_table(arrival_rate, num_samples, model_list, lat_lut, sampling_method='Poisson'):
  """
  Generates a set of input requests based on the supplied parameters. Each request 
  represents a task that consists of a <model, sample> pair - the model from the model list 
  and the sample from the target dataset. The sample affects sparsity and is implicitly sampled 
  from the latency LUT. Each request is assigned a random priority level.

  Args:
    arrival_rate: Mean arrival rate in tasks/s (or samples/s equivalently).
    num_samples: Total number of tasks/samples to generate.
    model_list: The candidate models for each task/sample.
    lat_lut: A LUT storing the latency measurements of the supplied models for the target accelerator.
    sampling_method: The distribution of the arrival process that determines the arrival times of the tasks/samples. 
  """
  reqst_table = []
  if sampling_method == 'Poisson': 
    # Each trial follows Exponential (Bernoulli) distribution, the counting results follow Poisson (Binominal)
    reqst_time = 0.0
    num_models = len(model_list)
    num_priority = len(PRIORITY_LIST)
    for i in range(num_samples):
      reqst_time += random.expovariate(arrival_rate)
      model_str = model_list[random.randint(0, num_models-1)] # Sample model, uniform sampling
      target_lat = lat_lut[model_str]['target_lat']
      avg_lat = lat_lut[model_str]['avg_lat'] # Used for PREMA time estimation
      priority = PRIORITY_LIST[random.randint(0, num_priority-1)] # Sample priority, uniform sampling
      num_examples = len(lat_lut[model_str])# Get the amount of data in the target dataset
      sample_id = sample_data(num_examples) # Sample from the dataset 
      avg_sparsity = lat_lut[model_str]['avg_sparsity']
      reqst_table.append([reqst_time, target_lat, model_str, priority, avg_lat, sample_id, avg_sparsity])
  else:
    raise NotImplementedError('Sampling approach not supoorted for request table construction.')
  return reqst_table



def construct_lat_table(models, csv_lat_files, args):
  """
  Construct a Look-Up Table (LUT) of latencies for the target accelerator based on the input csv files contain sparsity info
  
  Args:
    models: The set of models to benchmark.
    csv_lat_files: The per-layer level of sparsity of different samples 
      from a given dataset for the target model. This is generated using Sanger's codebase.
    args: Used to access the sequence length for Transformer models.
  """
  lat_lut = {}
  for n, model in enumerate(models):
    csv_lat_file = csv_lat_files[n]
    print ("Reading from ", csv_lat_file)
    metrics = pd.read_csv(csv_lat_file)
    sparsity = metrics['overall-sparsity']
    num_entries = len(sparsity)
    num_layers= np.max(metrics['layer-indx'])+1
    # Get the latency look-up table for each model
    load_balance = metrics['50%-skip']
    batch_latency_dict = {}
    batch_sparsity_dict = {}
    for i in range(num_entries):
      layer_lat = calc_sanger_latency(sparsity[i], load_balance[i], args.seq_len)
      if metrics['batch-indx'][i] not in batch_latency_dict:
        batch_latency_dict[metrics['batch-indx'][i]] = [ None for i in range(num_layers)]
        batch_sparsity_dict[metrics['batch-indx'][i]] = [ None for i in range(num_layers)]
      batch_latency_dict[metrics['batch-indx'][i]][metrics['layer-indx'][i]] = layer_lat
      batch_sparsity_dict[metrics['batch-indx'][i]][metrics['layer-indx'][i]] = sparsity[i]
      
    # Average latency per layer across different samples
    per_layer_latencies_avg = np.zeros(num_layers)
    for sample_idx,per_layer_latencies in batch_latency_dict.items():
      per_layer_latencies_avg = per_layer_latencies_avg + np.asarray(per_layer_latencies)

    num_samples = len(batch_latency_dict)
    per_layer_latencies_avg = np.divide(per_layer_latencies_avg, num_samples)

    # Average sparsity per layer across different samples
    per_layer_sparsity_avg = np.zeros(num_layers)
    for sample_idx,per_layer_sparsity in batch_sparsity_dict.items():
      per_layer_sparsity_avg = per_layer_sparsity_avg + np.asarray(per_layer_sparsity)
    
    num_samples = len(batch_sparsity_dict)
    per_layer_sparsity_avg = np.divide(per_layer_sparsity_avg, num_samples)


    # Get the target latency for each model 
    # The current method uses the mean, but can be extended to support others
    e2e_latency = []
    for k, v in batch_latency_dict.items(): # Accumulate all latency in each key
      e2e_latency.append(sum(batch_latency_dict[k]))

    # Get the variance of sparsity across layers for each model
    sparsity_var = []
    for sample_idx, sparsity in batch_sparsity_dict.items():
      var = np.var(sparsity)
      mean = np.mean(sparsity)
      sparsity_var.append(var/mean)

    # Draw the latency distribution 
    if (args.draw_dist):
      plt.hist(e2e_latency, density=True, bins=100)  # density=False would make counts
      plt.ylabel('Probability')
      plt.xlabel('Data')
      plt.savefig(os.path.join(args.figs_path, model+"_lat_dist.pdf"))
      plt.close()

      # Draw distbution of normalized variance of sparsity
      plt.hist(sparsity_var, density=True, bins=100)  # density=False would make counts
      plt.ylabel('Probability')
      plt.xlabel('Sparsity')
      plt.savefig(os.path.join(args.figs_path, model+"_sparsity_norm_dist.pdf"))
      plt.close()

    target_lat = np.mean(e2e_latency)

    # Insert into latency LUT 
    lat_lut[model] = {'lat_lut': batch_latency_dict, 'target_lat': target_lat, 'avg_lat': per_layer_latencies_avg,
                        'sparsity_lut': batch_sparsity_dict, 'avg_sparsity': per_layer_sparsity_avg}
  return lat_lut

def sample_data(num_examples):
  """
    Samples a random input from the target dataset. 
    This is to emulate random levels of sparsity.
    It currently uses uniform sampling, but it can be extended.
  """
  sample_id = random.randint(0, num_examples-1)
  return sample_id

class Task:
  """
  Represents an inference task, consisting of a model.

  Args:
    reqst_time: Arrival time of the request.
    target_lat: Target latency deadline.
    model_str: Target model.
    priority: Assigned priority level.
  """
  def __init__(self, reqst_time, target_lat, model_str, priority, avg_lat, avg_sparsity):
    # Common properties
    self.reqst_time = reqst_time
    self.target_time = self.reqst_time + target_lat # target end time
    self.real_isolated_time = -1 # Isolated time using real sparsity, initalize as -1, updated in construct_task()
    self.est_isolated_time = -1 # Isolated time without knowing real sparsity, initalize as -1, updated in construct_task()
    self.finish_time = -1 # initialize as -1
    self.model_str = model_str
    self.real_lat_queue = []
    self.priority = priority
    self.last_exe_time = self.reqst_time 
    self.est_lat_queue = list(avg_lat) # Estimated latency, initialized as the average latency

    # For Dysta use
    self.dysta_urgency = -1
    self.dysta_score = -1
    self.dysta_avg_sparsities = avg_sparsity
    self.real_sparsities = []
    self.dysta_measured_sparsities = []
    self.dysta_gamma = 1.0 # The ratio betten average and measured sparsity, used to estimate the latency queue

    # For SDRM use
    self.sdrm_urgency = -1
    self.sdrm_map_score = -1

    # For PREMA use
    self.prema_token = -1

    # For Planaria use
    self.planaria_score = -1

  def construct_task(self, lat_table, sparsity_table):
    """ 
    Adds all layers of the model to the task's queue.
    """
    for i in range(len(lat_table)):
      self.real_lat_queue.append(lat_table[i])
    
    for i in range(len(sparsity_table)):
      self.real_sparsities.append(sparsity_table[i])
    self.dysta_gamma = (1 - self.real_sparsities[0]) / (1 - self.dysta_avg_sparsities[0])
    self.real_isolated_time = sum(self.real_lat_queue)
    self.prema_est_isolated_time = sum(self.est_lat_queue)

  def exe(self, is_hw_monitor=False):
    """
    Executes the current layer.
    """
    lat = self.real_lat_queue.pop(0)
    self.est_lat_queue.pop(0)
    if (is_hw_monitor):
      sparsity = self.real_sparsities.pop(0)
      self.dysta_measured_sparsities.append(sparsity)
    return lat

  def is_finished(self, sys_time):
    """
    Checks whether all layers of the model 
    have been executed.
    """
    if (len(self.real_lat_queue) == 0):
      self.finish_time = sys_time
      return True
    else:
      return False