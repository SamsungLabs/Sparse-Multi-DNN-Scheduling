from scheduler import FCFS_Scheduler, Dysta_Scheduler, PREMA_Scheduler
from utils import generate_reqst_table, construct_lat_table
import sys
import argparse
import logging
import random
import matplotlib.pyplot as plt

BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN = "#2FC1D3"
BLUE = "#076FA1"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
RED = "#E3120B"

scheduler_dict = {"fcfs": FCFS_Scheduler,
                  "dysta": Dysta_Scheduler,
                  "prema_sparse": PREMA_Scheduler,
                  "prema": PREMA_Scheduler
                  }

def simulation(args):
  # Construct latency Look-Up Table (LUT)
  #   {"model_str": {"lat_lut": {{batch_no: [lat_layer1, lat_layer2, ...], batch_no: [lat_layer1, lat_layer2, ...]}}, "mean_lat": float}, ....}
  lat_lut = construct_lat_table(args.models, args.csv_lat_files, args)
  reqst_table_base = generate_reqst_table(args.sample_per_sec, args.num_sample, args.models, lat_lut)
 
  metrics = {"vio_rate":{},
            "thrpt":{},
            "antt":{}}
  # Evaluate each scheduler
  for lat_slo_mult in args.lat_slo_mult:
    # if (lat_slo_mult == 3):
    #   print ("enable debug mode")
    #   logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # else:
    #   logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    reqst_table = reqst_table_base.copy()
    # Update target target_lat using lat_slo_mult
    logging.info("*************** Creating new reqst tables using lat_slo_mult:%f ***************", lat_slo_mult)
    for reqst in reqst_table:
      reqst[1] = reqst[1] * lat_slo_mult # The second element reqst[1] is the target lat
      logging.debug("The target lat:%f" % (reqst[1]))
    # Handle request using different schedulers 
    for scheduler_name in args.schedule_method:
      print ("-"*100)
      if str.endswith(scheduler_name, "sparse"): scheduler = scheduler_dict[scheduler_name](reqst_table, is_sparse=True)
      else: scheduler = scheduler_dict[scheduler_name](reqst_table)
      scheduler.set_lat_lut(lat_lut)
      while (not scheduler.is_finished()):
        scheduler.update_reqst()
        scheduler.exe()

      # Get violation rate
      violation_rate, violate_task_list = scheduler.calc_violation_rate()
      if scheduler_name not in metrics["vio_rate"]: metrics["vio_rate"][scheduler_name] = [violation_rate]
      else: metrics["vio_rate"][scheduler_name].append(violation_rate)

      print ("Violation rate of ", scheduler_name, " scheduling:", violation_rate)
      for i in range(len(violate_task_list)):
        print ("Violate task:", violate_task_list[i])

      system_thrpt = scheduler.calc_system_thrpt()
      print(f"System Throughput (STP): {system_thrpt:.2f} inf/s")
      if scheduler_name not in metrics["thrpt"]: metrics["thrpt"][scheduler_name] = [system_thrpt]
      else: metrics["thrpt"][scheduler_name].append(system_thrpt)

      antt = scheduler.calc_ANTT()
      print(f"Average Normalised Turnaround Time (ANTT): {antt:.2f}")
      if scheduler_name not in metrics["antt"]: metrics["antt"][scheduler_name] = [antt]
      else: metrics["antt"][scheduler_name].append(antt)
      print ("-"*100)
  # Drawing figs using obtained metrics
  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 2.0))
  COLORS = [BLUE, GREEN, BROWN_DARKER, GREY, RED]
  tick_font_size = 9
  label_font_size = 13
  for i, (k, v) in enumerate(metrics.items()):
    metric_name = k
    # Get subplot
    ax = axs[i]
    for j, (s, results) in enumerate(v.items()):
      schedule_name = s
      ax.plot(args.lat_slo_mult, results, color=COLORS[j], lw=1.5, label=schedule_name)
    ax.set_xlabel('lat_slo_mult', fontsize = label_font_size)
    ax.set_ylabel(metric_name, fontsize = label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.grid()
    # ax.set_title(metric_name,y=0, pad=-53, fontsize = label_font_size)#, fontweight="bold"
    if i==0: 
        ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-1})
  fig.savefig("Metrics.pdf", bbox_inches='tight')
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  parser.add_argument("--models", nargs='+', default="bert", type=str, choices=["bert", "bart", "gpt2"],
                      help="The name(s) of candidate models.")
  parser.add_argument("--seq_len", default=512, type=int, required=False,
                      help="The input sequence length for Transformer models.")
  parser.add_argument("--csv_lat_files", nargs='+', default="bert_lat.csv", type=str,
                      help="The measured latencies of the supplied model(s) on the target accelerator.")
  parser.add_argument("--schedule_method", nargs='+', default="fcfs", type=str, choices=["fcfs", "dysta", "prema_sparse", "prema"],
                      help="The name(s) of the evaluated scheduling method(s).")
  parser.add_argument("--sample_per_sec", default=30, type=int, 
                      help="The input arrival rate in samples (or tasks) per second.")
  parser.add_argument("--num_sample", default=200, type=int, 
                      help="The total number of samples to simulate.")
  parser.add_argument("--lat_slo_mult", nargs='+', default=1.0, type=float,
                      help="Sets the target latency SLO as Mean Isolated Latency x SLO Multiplier (the supplied parameter). Typical values: 1.0 (unattainable), 10 (strict), 100 (loose).")
  # parser.add_argument("--lat_estimate_mean", action='store_true', default=False,
  #                     help="Use the mean per-layer latency as the latency estimate rather than the actual latency due to the variability in sparsity. Currently, it affects both the scheduling decisions and the accelerator remains idle.")
  parser.add_argument("--draw_dist", action="store_true")
  parser.add_argument("--figs_path", default="./", type=str, help="The path to all saved figures/images")

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
  if type(args.schedule_method) is not list:
      args.schedule_method = [args.schedule_method]
  if type(args.lat_slo_mult) is not list:
      args.lat_slo_mult = [args.lat_slo_mult]

  # Logging setup
  if args.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  random.seed(args.seed)

  simulation(args)