from scheduler import *
from utils_common import generate_reqst_table, construct_lat_table, calc_mean
from utils_figs import draw_figs
import sys
import argparse
import logging
import random
import matplotlib.pyplot as plt
import os
import copy

scheduler_dict = {"fcfs": FCFS_Scheduler,
                  "dysta_oracle": Dysta_Oracle_Scheduler,
                  "dysta": Dysta_Scheduler,
                  "dysta_naive": Dysta_Scheduler,
                  "prema_sparse": PREMA_Scheduler,
                  "prema": PREMA_Scheduler,
                  "sdrm3": SDRM3_Scheduler,
                  "sdrm3_sparse": SDRM3_Scheduler,
                  "sjf": SJF_Scheduler,
                  "sjf_sparse": SJF_Scheduler,
                  "planaria": Planaria, 
                  }

def simulation(args):
  # Construct latency Look-Up Table (LUT)
  #   {"model_str": {"lat_lut": {{batch_no: [lat_layer1, lat_layer2, ...], batch_no: [lat_layer1, lat_layer2, ...]}}, "mean_lat": float}, ....}
  lat_lut = construct_lat_table(args.models, args.csv_lat_files, args)
  ar_metrics = {} # arrival rate and metrics dict
  for r in range(args.num_random_seeds):
    random.seed(args.seed + r)
    for samples_per_sec in args.sample_per_sec:
      # Generate reqst table
      reqst_table_base = generate_reqst_table(samples_per_sec, args.num_sample, args.models, lat_lut)
      if (r == 0): # The frist iteration, create dict
        metrics = {"vio_rate":{}, "thrpt":{}, "antt":{}}
      else:
        metrics = ar_metrics[samples_per_sec]
      # Evaluate each scheduler
      for slo_indx, lat_slo_mult in enumerate(args.lat_slo_mult):
        reqst_table = copy.deepcopy(reqst_table_base)
        # Update target target_lat using lat_slo_mult
        logging.info("+++++++++++++++++++++ Experimenting with Arivval Rate:%f ++++++++++++++++++++++++", samples_per_sec)
        logging.info("*************** Creating new reqst tables using lat_slo_mult:%f ***************", lat_slo_mult)
        for reqst in reqst_table:
          reqst[1] = reqst[1] * lat_slo_mult # The second element reqst[1] is the target lat
          logging.debug("The reqst_time:%f, target lat:%f, model_str:%s" % (reqst[0], reqst[1], reqst[2]))
        # Handle request using different schedulers 
        for scheduler_name in args.schedule_method:
          print ("-"*100)
          if 'sdrm3' in scheduler_name: scheduler = scheduler_dict[scheduler_name](reqst_table, alpha=args.alpha, is_sparse=('sparse' in scheduler_name))
          elif 'dysta' in scheduler_name: scheduler = scheduler_dict[scheduler_name](reqst_table, args.vio_penalty_eff, args.num_candidate, args.beta, 'naive' not in scheduler_name)
          elif str.endswith(scheduler_name, "sparse"): scheduler = scheduler_dict[scheduler_name](reqst_table, is_sparse=True)
          else: scheduler = scheduler_dict[scheduler_name](reqst_table)
          scheduler.set_lat_lut(lat_lut)
          is_pattern_aware = ('naive' not in scheduler_name) and (('sparse' in scheduler_name) or ('dysta' in scheduler_name))
          while (not scheduler.is_finished()):
            scheduler.update_reqst(is_pattern_aware)
            scheduler.exe(is_hw_monitor=((scheduler_name == 'dysta') or ('sparse' in scheduler_name)))
          # Compute Violation Rate
          violation_rate, violate_task_dict = scheduler.calc_violation_rate()
          print ("Violation rate of ", scheduler_name, " scheduling:", violation_rate)
          if (args.print_vio_task):
            for k, v in violate_task_dict.items():
              print ("Violate task:", v)
          # Compute Throughput
          system_thrpt = scheduler.calc_system_thrpt()
          print(f"System Throughput (STP): {system_thrpt:.2f} inf/s")
          # Compute ANTT
          antt = scheduler.calc_ANTT()
          print(f"Average Normalised Turnaround Time (ANTT): {antt:.2f}")

          # Append results to metrics dict
          if (r == 0): # The frist round, create list
            # Append Violation Rate
            if scheduler_name not in metrics["vio_rate"]: metrics["vio_rate"][scheduler_name] = [[violation_rate]]
            else: metrics["vio_rate"][scheduler_name].append([violation_rate])
            # Append Throughput
            if scheduler_name not in metrics["thrpt"]: metrics["thrpt"][scheduler_name] = [[system_thrpt]]
            else: metrics["thrpt"][scheduler_name].append([system_thrpt])
            # Append ANTT
            if scheduler_name not in metrics["antt"]: metrics["antt"][scheduler_name] = [[antt]]
            else: metrics["antt"][scheduler_name].append([antt])
          else:
            metrics["vio_rate"][scheduler_name][slo_indx].append(violation_rate)
            metrics["thrpt"][scheduler_name][slo_indx].append(system_thrpt)
            metrics["antt"][scheduler_name][slo_indx].append(antt)
          print ("-"*100)
      ar_metrics[samples_per_sec] = metrics
  print (ar_metrics)
  ar_metrics = calc_mean(ar_metrics)
  # Drawing figs using obtained metrics
  draw_figs(args, ar_metrics)
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  parser.add_argument("--models", nargs='+', default="bert", type=str, choices=["bert", "bart", "gpt2", "mobilenet", "resnet50", "ssd",  "vgg16"],
                      help="The name(s) of candidate models.")
  parser.add_argument("--seq_len", default=512, type=int, required=False,
                      help="The input sequence length for Transformer models.")
  parser.add_argument("--csv_lat_files", nargs='+', default="bert_lat.csv", type=str,
                      help="The measured latencies of the supplied model(s) on the target accelerator.")
  parser.add_argument("--csv_lat_dir", default=None, type=str,
                      help="The directory of measured latency csv files.")
  parser.add_argument("--schedule_method", nargs='+', default="fcfs", type=str, choices=["fcfs", "dysta_oracle", "dysta", "dysta_naive", "prema_sparse", "prema", "sdrm3_sparse", "sdrm3", "sjf_sparse", "sjf", "planaria"],
                      help="The name(s) of the evaluated scheduling method(s).")
  parser.add_argument("--alpha", default=1.0, type=float, 
                      help="The fairness weight for SDRM3's MapScore metric.")
  parser.add_argument("--beta", default=0.01, type=float, 
                      help="The parameter used to control weighting of each metrics in Dysta scheduler")
  parser.add_argument("--sample_per_sec", nargs='+', default=30, type=int, 
                      help="The input arrival rate in samples (or tasks) per second.")
  parser.add_argument("--num_sample", default=30, type=int, 
                      help="The total number of samples to simulate.")
  parser.add_argument("--lat_slo_mult", nargs='+', default=1.0, type=float,
                      help="Sets the target latency SLO as Mean Isolated Latency x SLO Multiplier (the supplied parameter). Typical values: 1.0 (unattainable), 10 (strict), 100 (loose).")
  parser.add_argument("--vio_penalty_eff", default=1.0, type=float, 
                      help="Dysta parameter to control the effect of violation penalty.")
  parser.add_argument("--num_candidate", default=5, type=int, 
                      help="Dysta parameter to control the number of candidates after obtaining the score.")          
  parser.add_argument("--num_random_seeds", default=5, type=int, 
                      help="The number of random seeds to get geo mean results.")                
  parser.add_argument("--draw_dist", action="store_true")
  parser.add_argument("--fig_across_ar", action="store_true")
  parser.add_argument("--fig_across_slo", action="store_true")
  parser.add_argument("--fig_sparse_effect", action="store_true")
  parser.add_argument("--fig_tradeoff_analyse", action="store_true")
  
  
  parser.add_argument("--figs_path", default="./", type=str, help="The path to all saved figures/images")

  # Verbosity / Logging
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--print_vio_task", action="store_true")

  # Random seed
  parser.add_argument("--seed", type=int, default=1,
                      help="Random seed.")

  # parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  # parser.add_argument("--w_sparsity", default=0.95, type=float, help="Sparity of weight")

  args = parser.parse_args()
  if args.csv_lat_dir is not None:
    args.models = []
    args.csv_lat_files = []
    for csv_file in os.listdir(args.csv_lat_dir):
      if 'csv' not in csv_file: continue
      args.csv_lat_files.append(os.path.join(args.csv_lat_dir, csv_file))
      args.models.append(csv_file.split('.')[0])
  else:
    if type(args.models) is not list:
        args.models = [args.models]
    if type(args.csv_lat_files) is not list:
        args.csv_lat_files = [args.csv_lat_files]
  if type(args.schedule_method) is not list:
      args.schedule_method = [args.schedule_method]
  if type(args.lat_slo_mult) is not list:
      args.lat_slo_mult = [args.lat_slo_mult]
  if type(args.sample_per_sec) is not list:
      args.sample_per_sec = [args.sample_per_sec]

  # Logging setup
  if args.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  simulation(args)