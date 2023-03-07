import argparse
import logging
import sys

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