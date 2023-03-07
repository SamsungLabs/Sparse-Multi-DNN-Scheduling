from scheduler import *
from utils import *
import argparse
import sys
import numpy as np

logger = logging.getLogger()


scheduler_dict = {"fcfs": FCFS_Scheduler,
                  "dysta": Dysta_Scheduler,
                  "prema": PREAMA_Scheduler
                  }

def simulation(args):
  # Construct lut_lat
  # {"model_str": {"lat_lut": {{batch_no: [lat_layer1, lat_layer2, ...], batch_no: [lat_layer1, lat_layer2, ...]}}, "mean_lat": float}, ....}
  lat_lut = construct_lat_table(args.models, args.csv_lat_files, args)
  reqst_table = generate_reqst_table(args.sample_per_sec, args.sample_per_sec, args.models, lat_lut)
  for scheduler_name in args.schedule_method:
    scheduler = scheduler_dict[scheduler_name](reqst_table)
    scheduler.set_lat_lut(lat_lut)
    while (not scheduler.is_finish()):
      scheduler.update_reqst()
      scheduler.exe()

    # Get violation rate
    violation_rate, violate_task_list = scheduler.cal_violation_rate()
    print ("Violation rate of ", scheduler_name, " scheduling:", violation_rate)
    print ("+"*100)
    for i in range(len(violate_task_list)):
      print ("Violate task:", violate_task_list[i])
    print ("-"*100)
    print ("-"*100)
    



if __name__ == '__main__':
  # Let's allow the user to pass the filename as an argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--models", nargs='+', default="bert", type=str, help="The name of model candidats", choices=["bert", "bart"])
  parser.add_argument("--csv_lat_files", nargs='+', default="bert_lat.csv", type=str, help="The name of model candidats")
  parser.add_argument("--schedule_method", nargs='+', default="fcfs", type=str, help="The name of scheduling approach", choices=["fcfs", "dysta", "prema"])
  parser.add_argument("--sample_per_sec", default=30, type=int, help="The number of samples per second")
  parser.add_argument("--seq_len", default=512, type=int, required=False)
  parser.add_argument("--debug", action="store_true")

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

  if args.debug:
      logger.setLevel(logging.DEBUG) 
  else:
      logger.setLevel(logging.INFO)
  # assert len(args.models) == len(args.csv_lat_files)
  simulation(args)