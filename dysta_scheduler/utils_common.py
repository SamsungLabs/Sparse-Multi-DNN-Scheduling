import sys
import random
import pandas as pd
import numpy as np
import copy
from bench_sanger_v3 import calc_sanger_latency
import matplotlib.pyplot as plt
import os
from itertools import islice
from scipy.stats import gmean
from utils_pred import *

PRIORITY_LIST = [1, 3, 9] # PREMA's priority scheme

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

#########################################
############## Common Util ##############
#########################################

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
      target_lat = lat_lut[model_str]['target_lat_per_model']
      avg_lat = lat_lut[model_str]['avg_lat_per_model'] # latency per model, average across different patterns
      avg_lat_per_pattern = lat_lut[model_str]['avg_lat_per_pattern'] # latency per pattern per model
      priority = PRIORITY_LIST[random.randint(0, num_priority-1)] # Sample priority, uniform sampling
      num_examples = len(lat_lut[model_str])# Get the amount of data in the target dataset
      sample_id = sample_data(num_examples) # Sample from the dataset 
      avg_sparsity = lat_lut[model_str]['avg_sparsity']
      reqst_table.append([reqst_time, target_lat, model_str, priority, avg_lat, sample_id, avg_sparsity, avg_lat_per_pattern])
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
    if ('sanger' in csv_lat_file): load_balance = metrics['50%-skip']
    elif ('eyerissv2' in csv_lat_file): sim_lat = metrics['sim_lat'] 
    batch_latency_dict = {}
    batch_sparsity_dict = {}
    for i in range(num_entries):
      if ('sanger' in csv_lat_file): layer_lat = calc_sanger_latency(sparsity[i], load_balance[i], args.seq_len)
      elif ('eyerissv2' in csv_lat_file): layer_lat = sim_lat[i] 
      if metrics['batch-indx'][i] not in batch_latency_dict:
        batch_latency_dict[metrics['batch-indx'][i]] = [ None for i in range(num_layers)]
        batch_sparsity_dict[metrics['batch-indx'][i]] = [ None for i in range(num_layers)]
      batch_latency_dict[metrics['batch-indx'][i]][metrics['layer-indx'][i]] = layer_lat
      # Sanger sparsityp means the number of non-zeros
      batch_sparsity_dict[metrics['batch-indx'][i]][metrics['layer-indx'][i]] = 1 - sparsity[i] if ('sanger' in csv_lat_file) else sparsity[i] 
      
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
    lat_lut[model] = {'lat_lut': batch_latency_dict, 'target_lat_per_pattern': target_lat, 'avg_lat_per_pattern': per_layer_latencies_avg, 
                        'sparsity_lut': batch_sparsity_dict, 'avg_sparsity': per_layer_sparsity_avg}
  lat_lut = calc_avg_lat_per_model(lat_lut, args)
  return lat_lut

def calc_avg_lat_per_model(lut,args):
  """
    For each model, append average latency across sparsity patterns.
  """
  # For Sanger, because we did not consider different sparsity patterns, avg_lat_per_pattern equals to avg_lat_per_model
  if ('sanger' in args.csv_lat_files[0]):
    for model_name, info_dict in lut.items():
      info_dict['target_lat_per_model'] = info_dict['target_lat_per_pattern']
      info_dict['avg_lat_per_model'] = info_dict['avg_lat_per_pattern']
  else:
    lat_per_model = {}
    # Get latency information from lut
    for model_name, info_dict in lut.items():
      model_arch = model_name.split('_')[0]
      target_lat_per_pattern = lut[model_name]['target_lat_per_pattern']
      num_batch = len(lut[model_name]['lat_lut'])
      avg_lat_per_pattern = lut[model_name]['avg_lat_per_pattern']
      if model_arch not in lat_per_model:
        lat_per_model[model_arch] = {"num_batch" : [num_batch],
                                    "target_lat_per_pattern": [target_lat_per_pattern],
                                    "avg_lat_per_pattern": [avg_lat_per_pattern]}
      else:
        lat_per_model[model_arch]["num_batch"].append(num_batch)
        lat_per_model[model_arch]["target_lat_per_pattern"].append(target_lat_per_pattern)
        lat_per_model[model_arch]["avg_lat_per_pattern"].append(avg_lat_per_pattern)

    
    # Calculate per model latency
    for model_arch, info_dict_per_model in lat_per_model.items():
      num_batch_list = info_dict_per_model["num_batch"]
      target_lat_list = info_dict_per_model["target_lat_per_pattern"]
      avg_lat_list = info_dict_per_model["avg_lat_per_pattern"]
      target_lat_per_model = 0
      avg_lat_per_model = np.zeros(len(avg_lat_list[0]))
      total_batches = 0
      for i in range(len(target_lat_list)):
        total_batches += num_batch_list[i]
        target_lat_per_model += num_batch_list[i] * target_lat_list[i]
        avg_lat_per_model = avg_lat_per_model + num_batch_list[i] * avg_lat_list[i] 
      target_lat_per_model = target_lat_per_model / total_batches
      avg_lat_per_model = avg_lat_per_model / total_batches
      info_dict_per_model["target_lat_per_model"] = target_lat_per_model
      info_dict_per_model["avg_lat_per_model"] = avg_lat_per_model
    # Append per-model latency to the dict
    for model_name, info_dict in lut.items():
      model_arch = model_name.split('_')[0]
      info_dict["target_lat_per_model"] = lat_per_model[model_arch]["target_lat_per_model"]
      info_dict["avg_lat_per_model"] = lat_per_model[model_arch]["avg_lat_per_model"]
  
  # for model_name, info_dict in lut.items():
  #   print ("%s model per pattern %f, per model %f"%(model_name, info_dict["target_lat_per_pattern"], info_dict["target_lat_per_model"]))
  #   print (info_dict["avg_lat_per_model"])
  #   print (info_dict["avg_lat_per_pattern"])
  # sys.exit()
  return lut

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
  def __init__(self, reqst_time, target_lat, model_str, priority, avg_lat, avg_sparsity, avg_lat_per_pattern, is_pattern_aware=False):
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
    if (is_pattern_aware):
      self.est_lat_queue = list(avg_lat_per_pattern) # Estimated latency, initialized as the average latency
    else:
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
    # self.dysta_gamma = (1 - self.real_sparsities[0]) / (1 - self.dysta_avg_sparsities[0])
    self.dysta_gamma = 1.0
    self.real_isolated_time = sum(self.real_lat_queue)
    self.est_isolated_time = sum(self.est_lat_queue)

  def exe(self, is_hw_monitor=False):
    """
    Executes the current layer.
    """
    lat = self.real_lat_queue.pop(0)
    self.est_lat_queue.pop(0)
    if (is_hw_monitor):
      sparsity = self.real_sparsities.pop(0)
      self.dysta_measured_sparsities.append(sparsity)
    # Update gamma
    if (len(self.dysta_measured_sparsities) > 0):
      # Last one predictor because: (Check dysta_lat_pred.py for details.) 
      #     1. it has the lower RMSE
      #     2. Lower resource consumption (Momeory and computation).
      self.dysta_gamma = avg_pred_linear_rate(self.dysta_measured_sparsities, self.dysta_avg_sparsities)
      # self.dysta_gamma = last_N_pred_linear_rate(self.dysta_measured_sparsities, self.dysta_avg_sparsities)
      # self.dysta_gamma = last_one_pred_linear_rate(self.dysta_measured_sparsities, self.dysta_avg_sparsities)
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

def get_target_latency(lat_lut):
  """
  Calculate average E2E latency as target latency

  Args:
    lat_lut: Input latency LUT 
  """
  e2e_latency = []
  for k, v in lat_lut.items(): # Accumulate all latency in each key
    e2e_latency.append(sum(lat_lut[k]))
  target_lat = np.mean(e2e_latency)
  return target_lat

def cal_per_layer_avg(lut):
  """
  Calculate per-layer average value across difference data entries

  Args:
    lut: Input latency/sparsity LUT 
  """
  num_layers = len(lut[0])
  per_layer_avg = np.zeros(num_layers)
  for sample_idx,per_layer_v in lut.items():
    per_layer_avg = per_layer_avg + np.asarray(per_layer_v)

  num_samples = len(lut)
  per_layer_avg = np.divide(per_layer_avg, num_samples)
  return per_layer_avg

def split_lut(lut, train_ratio, test_ratio):
  """
  Split LUT into train and test sets.

  Args:
    lut: Input LUT contain all the data entries
    train_ratio: Ratio of tain data entries
    test_ratio: Ratio of test data entries 
  """
  assert (train_ratio + test_ratio) == 1.0
  train_lut = {}
  test_lut = {}
  # Each model in lut 
  for model, v_dict in lut.items():
    train_lat_lut = {}
    test_lat_lut = {}
    train_sparsity_lut = {}
    test_sparsity_lut = {}
    num_batch = len(v_dict['lat_lut'])
    train_indx = 0
    test_indx = 0
    # Get Latency Table
    for i in range(num_batch):
      if (random.uniform(0, 1)<=train_ratio):
        # Train set
        train_lat_lut[train_indx] = v_dict['lat_lut'][i]
        train_sparsity_lut[train_indx] = v_dict['sparsity_lut'][i]
        train_indx += 1
      else:
        # Train set
        test_lat_lut[test_indx] = v_dict['lat_lut'][i]
        test_sparsity_lut[test_indx] = v_dict['sparsity_lut'][i]
        test_indx += 1
    # Get End-to-end Latency
    train_target_lat = get_target_latency(train_lat_lut)
    test_target_lat = get_target_latency(test_lat_lut)

    # Get Per-layer Average
    train_per_layer_latencies_avg = cal_per_layer_avg(train_lat_lut)
    train_per_layer_sparsity_avg = cal_per_layer_avg(train_sparsity_lut)
    test_per_layer_latencies_avg = cal_per_layer_avg(test_lat_lut)
    test_per_layer_sparsity_avg = cal_per_layer_avg(test_sparsity_lut)
    train_lut[model] = {'lat_lut': train_lat_lut, 'target_lat': train_target_lat, 'avg_lat_per_pattern': train_per_layer_latencies_avg,
                        'sparsity_lut': train_sparsity_lut, 'avg_sparsity': train_per_layer_sparsity_avg}
    test_lut[model] = {'lat_lut': test_lat_lut, 'target_lat': test_target_lat, 'avg_lat_per_pattern': test_per_layer_latencies_avg,
                        'sparsity_lut': test_sparsity_lut, 'avg_sparsity': test_per_layer_sparsity_avg}
  return train_lut, test_lut

def calc_mean(metrics):
  """
  Get geo mean of metrics across different random seeds.

  Args:
    metrics: metrics dic with arrival rate as key, metric as values
  """
  print ("Calculating Geo Mean of results across different random seeds")
  for ar, ar_dict in metrics.items():
    for metric, metric_dict in ar_dict.items():
      for scheduler, result_list in metric_dict.items():
        # print (ar, metric, scheduler, result_list)
        mean_result = []
        for result in result_list:
          # mean_result.append(gmean(result))
          mean_result.append(np.mean(result))
        metrics[ar][metric][scheduler] = mean_result
        print ("arrival rate:%d, metric-%s:%f, scheduler:%s"%(ar, metric, mean_result[0], scheduler))
  return metrics
