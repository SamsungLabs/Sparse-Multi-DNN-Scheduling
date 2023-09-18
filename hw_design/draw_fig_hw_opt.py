import sys
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import argparse
import os

RESOURCE_NAMES = ('LUT', 'FF', 'DSP')#, 'BRAM')
HW_NAMES = ("Non_Opt_FP32", "Opt_FP32", "Opt_FP16")
RED_DEEP = "#CB0162"
GREEN = "#13BBAF"
ORANGE = "#FC824A"
BLUE = "#076FA1"
YELLOW = "#FDDC5C"
PINK_DEEP = "#D1768F"
BLUE_DARKER = "#02066F"
BLUE_LIGHT = "#CAFFFB"
BAR_COLORS = [[ORANGE, PINK_DEEP, RED_DEEP], [GREEN, BLUE, BLUE_DARKER]]

def extract_resource(file_name):
  """
  Extract the resource information from the target vivado report file

  Args:
    file_name: The name of target vivado report file
  """
  rpt_file = open(file_name, "r") 
  for line in rpt_file.readlines():
    if ("u_Dysta_Scheduler" in line) and ('(' not in line): 
      print ("Reading from vivado report %s"%(file_name))
      line = line.replace(' ', '')
      line = line.split('|')
      DSP_usage = int(line[-2])
      # BRAM_usage = int(line[-4])
      FF_usage = int(line[7])
      LUT_usage = int(line[3])
      return [LUT_usage, FF_usage, DSP_usage]#, BRAM_usage]

def draw_hw_opt(args):
  """
  Draw bar chart to demonstrate the effect of using different hardware optimizations

  Args:
    args: argumented parsed by main function
  """
  reqst_depths = [9, 6]
  norm_indx = 0
  max_depth = 0
  # Get resource utilzation from vivado rpt
  resc_utils = []
  for i, reqst_depth in enumerate(reqst_depths):
    rpt_suffix = "reqst" + str(reqst_depth) + ".rpt"
    resc_non_opt_fp32, resc_non_opt_fp16, resc_opt_fp16 = [], [], []
    for rpt_file in os.listdir(args.rpt_dir):
      if rpt_suffix not in rpt_file: continue
      rpt_path = os.path.join(args.rpt_dir, rpt_file)
      print (rpt_path)
      if "resc_util_non_opt_fp32" in rpt_file: resc_non_opt_fp32 = extract_resource(rpt_path)
      elif "resc_util_opt_fp32" in rpt_file: resc_non_opt_fp16 = extract_resource(rpt_path)
      elif "resc_util_opt_fp16" in rpt_file: resc_opt_fp16 = extract_resource(rpt_path)
      else: pass #raise RuntimeError('format of report file name is wroing')
    if max_depth < reqst_depth:
      max_depth = reqst_depth
      norm_indx = i
    print (resc_non_opt_fp32)
    print (resc_non_opt_fp16)
    print (resc_opt_fp16)
    resc_utils.append([resc_non_opt_fp32, resc_non_opt_fp16, resc_opt_fp16])
  
  # Normalize using the resource utilization with larger request depth and naive implementation
  norm_resc_utils = []
  for i in range(len(resc_utils)):
    norm_resc_non_opt_fp32 = [resc_utils[i][0][j]/resc_utils[i][0][j] for j in range(len(resc_utils[i][0]))]
    norm_resc_non_opt_fp16 = [resc_utils[i][1][j]/resc_utils[i][0][j] for j in range(len(resc_utils[i][1]))]
    norm_resc_opt_fp16 = [resc_utils[i][2][j]/resc_utils[i][0][j] for j in range(len(resc_utils[i][2]))]
    # norm_resc_non_opt_fp32 = [resc_utils[i][0][j]/resc_utils[norm_indx][0][j] for j in range(len(resc_utils[i][0]))]
    # norm_resc_non_opt_fp16 = [resc_utils[i][1][j]/resc_utils[norm_indx][0][j] for j in range(len(resc_utils[i][1]))]
    # norm_resc_opt_fp16 = [resc_utils[i][2][j]/resc_utils[norm_indx][0][j] for j in range(len(resc_utils[i][2]))]
    norm_resc_utils.append([norm_resc_non_opt_fp32, norm_resc_non_opt_fp16, norm_resc_opt_fp16])

  barWidth = 0.22
  tick_font_size = 9
  label_font_size = 13
  axis_idx = 0
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.0))
  indx = np.arange(len(RESOURCE_NAMES))
  for norm_resc_util in norm_resc_utils:
    ax = axs[axis_idx]
    for i in range(len(norm_resc_util)):
      ax.bar(
        indx + i*barWidth, 
        norm_resc_util[i], 
        color=BAR_COLORS[axis_idx][i], 
        edgecolor='black', 
        width=barWidth, 
        align='center', 
        label=HW_NAMES[i])
    ax.set_xticks(indx + (len(HW_NAMES) -1)*barWidth / 2, RESOURCE_NAMES)
    ax.legend(ncol=len(HW_NAMES), loc='lower left', bbox_to_anchor=(-0.06, 1.0), prop={'size': tick_font_size})
    ax.set_ylabel("Normalized Resource Usage", fontsize = label_font_size)
    ax.set_xlabel("Requst Depth " + str(1<<reqst_depths[axis_idx]), fontsize = label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.grid()
    ax.set_axisbelow(True)
    axis_idx += 1
  fig.tight_layout()
  fig.savefig("Hardware_Opt_Effect_Reqst.pdf", bbox_inches='tight')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simulator for scheduling sparse multi-DNN workloads on sparse DNN accelerators.")

  # Simulation configuration
  # parser.add_argument("--reqst_depth", default=6, type=int, choices=[6, 9],
  #                     help="The depth of request FIFO. The maximal number of requests is bounded by 2^reqst_depth")
  parser.add_argument("--rpt_dir", default='./vivado_rpt/', type=str,
                      help="The directory of measured latency csv files.")
  # parser.add_argument("--draw_dist", action="store_true")

  args = parser.parse_args()

  draw_hw_opt(args)