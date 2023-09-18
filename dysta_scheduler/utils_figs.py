
import sys
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

#########################################
############## Fig Drawing ##############
#########################################
BROWN = "#AD8C97"
BROWN_DARKER = "#7d3a46"
GREEN_LIGHT = "#2FC1D3"
GREEN_DARKER = "#048243"
BLUE = "#076FA1"
BLUE_LIGHT = "#CAFFFB"
BLUE_DARKER = "#02066F"
BLACK = "#1B2431"
GREY = "#C7C9CB"
GREY_DARKER = "#5C5B5D"
GREY_LIGHT = "#D8DCD6"
PURPLE = "#6D5ACF"
PURPLE_LIGHT = "#8F8CE7"
MERLOT = "#730039"

RED_LIGHT = "#FF6163"
RED = "#D3494E"
RED_DEEP = "#CB0162"
GREEN = "#13BBAF"
ORANGE = "#FC824A"
YELLOW = "#FDDC5C"
PINK = "#FF9A8A"
PINK_DEEP = "#D1768F"
MINT = "#47C072"

scheduler_names = {"fcfs": "FCFS",
                  "dysta_oracle": "Oracle",
                  "dysta": "Dysta",
                  "dysta_naive": "Dysta-naive",
                  "prema_sparse": "PREMA-sparse",
                  "prema": "PREMA",
                  "sdrm3": "SDRM3",
                  "sdrm3_sparse": "SDRM3-sparse",
                  "sjf": "SJF",
                  "sjf_sparse": "SJF-sparse",
                  "planaria": "Planaria", 
                  }

metric_names = {"vio_rate": "SLO Violation Rate (%)",
                "thrpt": "Throughput (inf/s)",
                "antt": "ANTT"
                }

scheduler_colors_attnn = {"fcfs": MERLOT,
                  "dysta_oracle": BLACK,
                  "dysta": GREY_DARKER,
                  "dysta_naive": GREY_LIGHT,
                  "prema_sparse": BROWN_DARKER,
                  "prema": BROWN,
                  "sdrm3": BLUE,
                  "sdrm3_sparse": BLUE_LIGHT,
                  "sjf": PURPLE_LIGHT,
                  "sjf_sparse": PURPLE,
                  "planaria": GREEN_LIGHT, 
                  }

scheduler_colors_cnn = {"fcfs": ORANGE,
                  "dysta_oracle": RED_DEEP,
                  "dysta": RED,
                  "dysta_naive": RED_LIGHT,
                  "prema_sparse": GREEN_DARKER,
                  "prema": GREEN,
                  "sdrm3": MINT,
                  "sdrm3_sparse": MINT, # TO-ADD
                  "sjf": PINK,
                  "sjf_sparse": PINK_DEEP,
                  "planaria": YELLOW, 
                  }

def draw_figs(args, ar_metrics):
  if args.fig_across_ar:
    draw_fig_across_ar(args, ar_metrics)
  if args.fig_across_slo:
    draw_fig_across_slo(args, ar_metrics)
  if args.fig_sparse_effect:
    draw_fig_sparse_effect(args, ar_metrics)
  if args.fig_tradeoff_analyse:
    draw_fig_tradeoff_analyse(args, ar_metrics)

def draw_fig_across_ar(args, ar_metrics):
  """
  Draw violation rate, throuput and antt across different arrival rate (ar)

  Args:
    args: argumented parsed by main function
    ar_metrics: metrics dictionary, e.g. {30: metrics_dict_of_30ar, 60: metrics_dict_of_60ar, ....}
  """
  prefix = None
  if ('sanger' in args.csv_lat_files[0]): 
    prefix = 'Sanger_'
    scheduler_colors = scheduler_colors_attnn
  elif ('eyerissv2' in args.csv_lat_files[0]): 
    prefix = 'EyerissV2_'
    scheduler_colors = scheduler_colors_cnn

  for slo_indx, lat_slo_mult in enumerate(args.lat_slo_mult):
    # Drawing figs using obtained metrics
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 2.0))
    tick_font_size = 9
    label_font_size = 13
    axis_idx = 0
    for i, (metric_name, label) in enumerate(metric_names.items()):

      # Get subplot
      ax = axs[axis_idx]
      for scheduler_name in args.schedule_method:
        # Get metrics from ar_metrics
        metric_list = []
        for samples_per_sec in args.sample_per_sec:
          metric_list.append(ar_metrics[samples_per_sec][metric_name][scheduler_name][slo_indx])
        if 'vio_rate' in metric_name:
          metric_list = [metric * 100 for metric in metric_list]
        ax.plot(
          args.sample_per_sec, 
          metric_list, 
          color=scheduler_colors[scheduler_name], 
          lw=1.5, 
          label=scheduler_name, 
          marker='o',
          )

      ax.set_xlabel('Arrival Rate (samples/s)', fontsize = label_font_size)
      ylabel_name = metric_names[metric_name]
      ax.set_ylabel(ylabel_name, fontsize = label_font_size)
      ax.tick_params(axis='both', labelsize=tick_font_size)
      ax.grid()
      # ax.set_title(metric_name,y=0, pad=-53, fontsize = label_font_size)#, fontweight="bold"
      if axis_idx==0: 
        label_names = [scheduler_names[schedule_name] for schedule_name in args.schedule_method]
        ax.legend(label_names, ncol=len(label_names), loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-1})
      axis_idx += 1

    fig.savefig(prefix + "Metrics_rate" + str(args.sample_per_sec) + "_slo_" + str(lat_slo_mult)
                 + "_sample" + str(args.num_sample) + "_across_arrival_rates.pdf", bbox_inches='tight')


def draw_fig_across_slo(args, ar_metrics):
  """
  Draw violation rate and antt across different SLOs

  Args:
    args: argumented parsed by main function
    ar_metrics: metrics dictionary, e.g. {30: metrics_dict_30ar, 60: metrics_dict_60ar, ....}
  """
  prefix = None
  if ('sanger' in args.csv_lat_files[0]): 
    prefix = 'Sanger_'
    scheduler_colors = scheduler_colors_attnn
  elif ('eyerissv2' in args.csv_lat_files[0]): 
    prefix = 'EyerissV2_'
    scheduler_colors = scheduler_colors_cnn

  for samples_per_sec in args.sample_per_sec:
    metrics = ar_metrics[samples_per_sec]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 2.0))
    tick_font_size = 9
    label_font_size = 13
    axis_idx = 0
    for i, (k, v) in enumerate(metrics.items()):
      metric_name = k
      
      # Skip system throughput
      if 'thrpt' in metric_name:
        continue

      # Get subplot
      ax = axs[axis_idx]
      for j, (s, results) in enumerate(v.items()):
        if 'vio_rate' in metric_name:
          results = [result*100 for result in results]
        schedule_name = scheduler_names[s] 
        ax.plot(
          args.lat_slo_mult, 
          results, 
          color=scheduler_colors[s], 
          lw=1.5, 
          label=schedule_name, 
          marker='o',
          )

      ax.set_xlabel('Latency SLO multiplier', fontsize = label_font_size)
      ylabel_name = metric_names[metric_name]
      ax.set_ylabel(ylabel_name, fontsize = label_font_size)
      ax.tick_params(axis='both', labelsize=tick_font_size)
      ax.grid()
      # ax.set_title(metric_name,y=0, pad=-53, fontsize = label_font_size)#, fontweight="bold"
      if axis_idx==0: 
        label_names = [scheduler_names[schedule_name] for schedule_name in args.schedule_method]
        ax.legend(ncol=len(label_names), loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-1})
      axis_idx += 1
    fig.savefig(prefix + "Metrics_rate" + str(samples_per_sec) + "_sample" + 
                  str(args.num_sample) + "_across_slo.pdf", bbox_inches='tight')


scheduler_names_sparse = {"dysta": "Dysta",
                        "dysta_naive": "Dysta-w/o-sparse",
                        "prema": "PREMA"
                        }

metric_color_att = {"vio_rate": "#2FC1D3",
                    "antt": "#076FA1",}
metric_color_cnn = {"vio_rate": PINK_DEEP,
                    "antt": RED_DEEP}
def draw_fig_sparse_effect(args, ar_metrics):
  """
  Draw bar chart to demonstrate the effect of using sparse latency predictor

  Args:
    args: argumented parsed by main function
    ar_metrics: metrics dictionary, e.g. {30: metrics_dict_30ar, 60: metrics_dict_60ar, ....}
  """
  prefix = None
  if ('sanger' in args.csv_lat_files[0]): 
    prefix = 'Sanger_'
    metric_color = metric_color_att
  elif ('eyerissv2' in args.csv_lat_files[0]): 
    prefix = 'EyerissV2_'
    metric_color = metric_color_cnn
  assert len(args.schedule_method) == 3
  barWidth = 0.37
  for samples_per_sec in args.sample_per_sec:
    for slo_indx, slo_mult in enumerate(args.lat_slo_mult):
      metrics = ar_metrics[samples_per_sec]
      tick_font_size = 9
      label_font_size = 12
      axis_idx = 0
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.0, 2.5))
      for i, (k, v) in enumerate(metrics.items()):
        metric_name = k
        # Skip system throughput
        if 'thrpt' in metric_name:
          continue
        # Get subplot
        ax = axs[axis_idx]
        if metric_name == "antt":
          ymax = -1
          for s, results in v.items():
            ymax = ymax if ymax > results[slo_indx] else results[slo_indx]
          ax.set_ylim(ymax*0.8, ymax*1.05)
        for j, (s, results) in enumerate(v.items()):
          schedule_name = scheduler_names_sparse[s] 
          y_value = results[slo_indx] * 100 if metric_name == "vio_rate" else results[slo_indx]
          ax.bar(
            [schedule_name], 
            [y_value], 
            color=metric_color[metric_name], 
            edgecolor='black', 
            width=barWidth, 
            align='center', 
            label=metric_name)
        ylabel_name = metric_names[metric_name]
        ax.set_ylabel(ylabel_name, fontsize = label_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.tick_params('x', labelrotation=15)
        ax.grid()
        ax.set_axisbelow(True)
        axis_idx += 1
      fig.tight_layout()
      fig.savefig(prefix + "Sparsity_Effect" + str(samples_per_sec) + "_sample" + 
            str(args.num_sample) + "_across_slo"+ str(slo_mult) + "_" + args.schedule_method[0] +".pdf", bbox_inches='tight')



ar_color_att = {20: BLUE_DARKER,
                30: BLUE_DARKER,
                40: BLUE_DARKER,}

ar_color_cnn = {3: RED,
                5: RED,
                10: RED,}

scheduler_mark = {"fcfs": "v",
                  "dysta": "*",
                  "prema": "s",
                  "sdrm3": "h",
                  "sjf": "P",
                  "planaria": "o", 
                  }
def draw_fig_tradeoff_analyse(args, ar_metrics):
  """
  Draw scatter plot to show the trade off of different scheduler in Viorate-ANTT 2D plot

  Args:
    args: argumented parsed by main function
    ar_metrics: metrics dictionary, e.g. {30: metrics_dict_30ar, 60: metrics_dict_60ar, ....}
  """
  prefix = None
  if ('sanger' in args.csv_lat_files[0]): 
    prefix = 'Sanger_'
    scheduler_colors = scheduler_colors_attnn
    ar_color = ar_color_att
    titles = ["(a) Multi-AttNNs with 30 samples/s", "(b) Multi-AttNNs with 40 samples/s"]
  elif ('eyerissv2' in args.csv_lat_files[0]): 
    prefix = 'EyerissV2_'
    scheduler_colors = scheduler_colors_cnn
    ar_color = ar_color_cnn
    titles = ["(c) Multi-CNNs with 3 samples/s", "(d) Multi-CNNs with 4 samples/s"]
  y_metric = "antt"
  x_metric = "vio_rate"
  label_names = [scheduler_names[schedule_name] for schedule_name in args.schedule_method]
  for slo_indx, slo_mult in enumerate(args.lat_slo_mult):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.5))
    for rate_indx, samples_per_sec in enumerate(args.sample_per_sec):
      # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 3.0))
      ax = axs[rate_indx]
      metrics = ar_metrics[samples_per_sec]
      tick_font_size = 9
      label_font_size = 13
      for schedule_name in args.schedule_method:
        x_value = metrics[x_metric][schedule_name][slo_indx]
        y_value = metrics[y_metric][schedule_name][slo_indx]
        ax.scatter(
          x_value * 100, 
          y_value, # Convert to percentage  
          c=scheduler_colors[schedule_name],
          #c=ar_color[samples_per_sec], 
          edgecolor="black", 
          marker=scheduler_mark[schedule_name],
          alpha=0.9,
          s=180 if schedule_name != "dysta" else 235,
          label=scheduler_names[schedule_name])
      # ax.legend(ncol=len(args.schedule_method), loc='lower left', bbox_to_anchor=(0.0, 1.0), prop={'size': label_font_size-2})
      ylabel_name = metric_names[y_metric]
      ax.set_ylabel(ylabel_name, fontsize = label_font_size)
      xlabel_name = metric_names[x_metric]
      ax.set_xlabel(xlabel_name, fontsize = label_font_size)
      ax.tick_params(axis='both', labelsize=tick_font_size)
      ax.set_title(titles[rate_indx], fontsize = label_font_size+3, y=-0.6)
      ax.grid()
      ax.set_axisbelow(True)
      # fig.tight_layout()
      # fig.savefig(prefix + "Tradeoff_slo"+ str(slo_mult) + "_sample" + str(samples_per_sec) + ".pdf", bbox_inches='tight')
      if (rate_indx == 0): fig.legend(label_names, ncol=len(label_names), loc='lower left', bbox_to_anchor=(0.07, 1.0), prop={'size': label_font_size})
    fig.tight_layout()
    fig.savefig(prefix + "Tradeoff_slo"+ str(slo_mult) + ".pdf", bbox_inches='tight')

