from scheduler import FCFS_Scheduler, Dysta_Scheduler, PREMA_Scheduler
from utils import generate_reqst_table, construct_lat_table
from options import args
import random
import logging

random.seed(args.seed)

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
  simulation(args)