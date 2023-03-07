import sys
import random
import pandas as pd
import numpy as np
from bench_sanger_v3 import calc_sanger_latency

PRIORITY_LIST = [1, 3, 9] # PREMA's priority scheme

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
    raise NotImplementedError('Sampling approach not supoorted for request table construction.')
  # print (reqst_table)
  return reqst_table

def construct_lat_table(models, csv_lat_files, args):
  """
  Populates a Look-Up Table (LUT) of latencies for the target accelerator.
  
  Args:
    models: The set of models to benchmark.
    csv_lat_files: The per-layer level of sparsity of different samples 
      from a given dataset for the target model. This is generated using Sanger's codebase.
    args: Used to access the sequence length for Transformer models.
  """
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
    # Get the target latency for each model 
    # The current method uses the mean, but can be extended to support others
    e2e_latency = []
    for k, v in batch_latency_dict.items(): # Accumulate all latency in each key
        e2e_latency.append(sum(batch_latency_dict[k]))
    target_lat = np.mean(e2e_latency)

    # Insert into latency LUT 
    lat_lut[model] = {'lat_lut': batch_latency_dict, 'target_lat': target_lat}
  return lat_lut

class Task:
  """
  Represents an inference task, consisting of a model, 
  a request arrival time, a target latency and a priority.

  Args:
    reqst_time: Arrival time of the request.
    target_lat: Target latency deadline.
    model_str: Target model.
    priority: Assigned priority level.
  """
  def __init__(self, reqst_time, target_lat, model_str, priority):
    self.reqst_time = reqst_time
    self.target_time = self.reqst_time + target_lat # target end time
    self.isolated_time = target_lat # TODO
    self.finish_time = -1 # initialize as -1
    self.model_str = model_str
    self.lat_queue = []
    self.priority = priority # initialize as -1
    self.urgency = -1
    self.prema_last_exe_time = self.reqst_time # For PREMA use
    self.prema_token = -1 # For PREMA use

  def sample_data(self, num_examples):
    """
      Samples a random input from the target dataset. 
      This is to emulate random levels of sparsity.
      It currently uses uniform sampling, but it can be extended.
    """
    sample_id = random.randint(0, num_examples-1)
    return sample_id

  def construct_task(self, lat_table):
    """
    Adds all layers of the model to the task's queue.
    """
    for i in range(len(lat_table)):
      self.lat_queue.append(lat_table[i])

  def exe(self):
    """
    Executes the current layer.
    """
    lat = self.lat_queue.pop(0)
    return lat

  def is_finished(self, sys_time):
    """
    Checkes whether all layers of the model 
    have been executed.
    """
    if (len(self.lat_queue) == 0):
      self.finish_time = sys_time
      return True
    else:
      return False