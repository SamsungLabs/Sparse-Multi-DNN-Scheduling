from scheduler import *
from utils import generate_reqst_table, construct_lat_table
import sys
import argparse
import logging
import random
import matplotlib.pyplot as plt
import os
import copy
import math

COLAR_MAPS = ['inferno', 'bone', 'viridis', 'magma', 'cividis']
FIG_INDX = ['(a)', '(b)', '(c)', '(d)', '(d)']

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


def lat_pred(args):
  task_lut = construct_lat_table(args.models, args.csv_lat_files, args)
  num_models = len(args.models)
  fig, axs = plt.subplots(nrows=1, ncols=num_models, figsize=(6*num_models, 5.0))
  if (args.draw_corr_matrix):
    for n, (model_key, v) in enumerate(task_lut.items()):
      ax = axs[n]
      batch_sparsity_dict = v['sparsity_lut']
      num_layers = len(batch_sparsity_dict[0])

      # Analyse corelation of sparsity among different layers
      layer_sparsity_list = []
      num_batch = len(batch_sparsity_dict)
      for i in range(num_layers):
        layer_sparsity = []
        for k, v in batch_sparsity_dict.items():
          layer_sparsity.append(v[i])
        layer_sparsity_list.append(layer_sparsity)
      corr_matrix = np.corrcoef(layer_sparsity_list)
      # print ("num_layer:", num_layers)
      # print ("corr_matrix:", corr_matrix)

      # Draw the correlation matrix among the sparsity in different layers
      colormap = plt.cm.get_cmap(COLAR_MAPS[n])
      sm = plt.cm.ScalarMappable(cmap=colormap)
      font_size=13
      label_size=13
      ax.matshow(corr_matrix, vmin=0.0, vmax=1.0, cmap=colormap)
      ax.set_ylabel('Correlation', fontsize=font_size)
      ax.set_xlabel('Layer Indx', fontsize=font_size)
      ax.set_title(FIG_INDX[n] + ' Correlation of Sparsity inzz ' + model_key, y = -0.2, fontsize=font_size+2)
      ax.tick_params(axis='x', labelsize=label_size)
      ax.tick_params(axis='y', labelsize=label_size)
      ax.set_xticks(range(len(corr_matrix)), rotation=45)
      ax.set_yticks(range(len(corr_matrix)))
      # ax.set_xticks(range(len(corr_matrix)), fontsize=font_size, rotation=45)
      # ax.set_yticks(range(len(corr_matrix)), fontsize=font_size)
      cbar = plt.colorbar(sm, ax=ax)
      cbar.ax.tick_params(labelsize=label_size)
    plt.savefig(os.path.join(args.figs_path, "sparsity_corr_matrix.pdf"))

  # Calculate the accuracy of Dysta's lat predictor
  for n, (model_key, v) in enumerate(task_lut.items()):
    for predictor in ['avg', 'lastN', 'last_one']:
      batch_sparsity_dict = copy.deepcopy(v['sparsity_lut'])
      batch_latency_dict = copy.deepcopy(v['lat_lut'])
      avg_sparsity_list = copy.deepcopy(v['avg_sparsity'])
      avg_lat_list = copy.deepcopy(v['avg_lat'])
      sum_abs_error_lat = 0
      sum_rel_error_lat = 0
      num_samples = 0
      sum_mse = 0 # Mean-square error
      for batch_no, real_sparsity in batch_sparsity_dict.items():
        num_layers = len(real_sparsity)
        measured_sparsities = [real_sparsity.pop(0)]
        for i in range(1, num_layers):
          if predictor == 'avg':
            linear_rate = avg_pred_linear_rate(measured_sparsities, avg_sparsity_list)
          elif predictor == 'lastN':
            linear_rate = last_N_pred_linear_rate(measured_sparsities, avg_sparsity_list)
          else:
            linear_rate = last_one_pred_linear_rate(measured_sparsities, avg_sparsity_list)
          real_lat = batch_latency_dict[batch_no][i]
          est_lat = avg_lat_list[i] * linear_rate
          abs_error_lat = abs(est_lat - real_lat)
          sum_abs_error_lat += abs_error_lat

          rel_error_lat = abs_error_lat / real_lat
          sum_rel_error_lat += rel_error_lat

          sum_mse += abs_error_lat * abs_error_lat
          num_samples += 1
          measured_sparsities.append(real_sparsity.pop(0))

      avg_abs_error = sum_abs_error_lat / num_samples
      avg_rel_error = sum_rel_error_lat / num_samples
      mse = sum_mse / num_samples
      rmse = math.sqrt(mse) # Root-mean-square error
      print ("Averaged abs error of %s with %s predictor: abs error %f, rel error %f, mse %f, rmse %f"% (model_key, predictor, avg_abs_error, avg_rel_error, mse, rmse))
    

      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  parser.add_argument("--models", nargs='+', default="bert", type=str, choices=["bert", "bart", "gpt2"],
                      help="The name(s) of candidate models.")
  parser.add_argument("--csv_lat_files", nargs='+', default="bert_lat.csv", type=str,
                      help="The measured latencies of the supplied model(s) on the target accelerator.")
  parser.add_argument("--figs_path", default="./", type=str, help="The path to all saved figures/images")
  parser.add_argument("--seq_len", default=512, type=int, required=False,
                      help="The input sequence length for Transformer models.")
  parser.add_argument("--draw_dist", action="store_true")
  parser.add_argument("--draw_corr_matrix", action="store_true")
  # Verbosity / Logging
  parser.add_argument("--debug", action="store_true")

  # Random seed
  parser.add_argument("--seed", type=int, default=1,
                      help="Random seed.")

  # parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  # parser.add_argument("--w_sparsity", default=0.95, type=float, help="Sparity of weight")

  args = parser.parse_args()
  if type(args.models) is not list:
      args.models = [args.models]
  if type(args.csv_lat_files) is not list:
      args.csv_lat_files = [args.csv_lat_files]


  # Logging setup
  if args.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  random.seed(args.seed)

  lat_pred(args)