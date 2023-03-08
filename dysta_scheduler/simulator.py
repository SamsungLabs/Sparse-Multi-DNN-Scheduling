from scheduler import FCFS_Scheduler, Dysta_Scheduler, PREMA_Scheduler
from utils import generate_reqst_table, construct_lat_table
import sys
import argparse
import logging
import random

scheduler_dict = {"fcfs": FCFS_Scheduler,
                  "dysta": Dysta_Scheduler,
                  "prema": PREMA_Scheduler
                  }

def simulation(args):
  # Construct latency Look-Up Table (LUT)
  #   {"model_str": {"lat_lut": {{batch_no: [lat_layer1, lat_layer2, ...], batch_no: [lat_layer1, lat_layer2, ...]}}, "mean_lat": float}, ....}
  lat_lut = construct_lat_table(args.models, args.csv_lat_files, args)
  reqst_table = generate_reqst_table(args.sample_per_sec, args.sample_per_sec, args.models, lat_lut)
 
  # Evaluate each scheduler
  for scheduler_name in args.schedule_method:
    scheduler = scheduler_dict[scheduler_name](reqst_table)
    scheduler.set_lat_lut(lat_lut)
    while (not scheduler.is_finished()):
      scheduler.update_reqst()
      scheduler.exe()

    # Get violation rate
    violation_rate, violate_task_list = scheduler.calc_violation_rate()
    print ("Violation rate of ", scheduler_name, " scheduling:", violation_rate)
    print ("+"*100)
    for i in range(len(violate_task_list)):
      print ("Violate task:", violate_task_list[i])
    print ("-"*100)
    print ("-"*100)

    system_thrpt = scheduler.calc_system_thrpt()
    print(f"System Throughput (STP): {system_thrpt:.2f} inf/s")

    antt = scheduler.calc_ANTT()
    print(f"Average Normalised Turnaround Time (ANTT): {antt:.2f}")
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  parser.add_argument("--models", nargs='+', default="bert", type=str, choices=["bert", "bart"],
                      help="The name(s) of candidate models.")
  parser.add_argument("--csv_lat_files", nargs='+', default="bert_lat.csv", type=str,
                      help="The measured latencies of the supplied model(s) on the target accelerator.")
  parser.add_argument("--schedule_method", nargs='+', default="fcfs", type=str, choices=["fcfs", "dysta", "prema"],
                      help="The name(s) of the evaluated scheduling method(s).")
  parser.add_argument("--sample_per_sec", default=30, type=int, 
                      help="The input arrival rate in samples (or tasks) per second.")
  parser.add_argument("--seq_len", default=512, type=int, required=False,
                      help="The input sequence length for Transformer models.")

  # Verbosity / Logging
  parser.add_argument("--debug", action="store_true")

  # Random seed
  parser.add_argument("--seed", type=int, default=1,
                      help="Random seed.")

  # parser.add_argument("--dataset_root", default="/path/to/your/directory", type=str, help="The path to your dataset root")
  # parser.add_argument("--figs_path", default="/path/to/your/directory", type=str, help="The path to all saved figures/images")
  # parser.add_argument("--w_sparsity", default=0.95, type=float, help="Sparity of weight")

  args = parser.parse_args()
  if type(args.models) is not list:
      args.models = [args.models]
  if type(args.csv_lat_files) is not list:
      args.csv_lat_files = [args.csv_lat_files]
  if type(args.schedule_method) is not list:
      args.schedule_method = [args.schedule_method]

  # Logging setup
  if args.debug:
      logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  else:
      logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  random.seed(args.seed)

  simulation(args)