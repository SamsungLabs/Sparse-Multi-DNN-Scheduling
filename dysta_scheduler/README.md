# Simulation of scheduling

Please install the environment as instructed in the top-level directory.

## Run on Sanger

1. Follow the instruction of README under csv_files to download csv files.
2. Run the following command to compare FCFS with PREMA.
```
bash script/sanger_multisparse_scheduling.sh
# you can also enable the debug info, or draw lat distbution
python simulator.py --schedule_method fcfs prema --debug --draw_dist
```

## Evaluate Dysta Scheduling
Run the following script:
```
bash script/dysta_comparison_sanger_w_pred.sh
```
You can check the generate pdf `Metrics_rate**_sample**.pdf`

## Evaluate Dysta Latency Predictor
Run the following script:
```
bash script/dysta_lat_pred_draw_correlation.sh 
```
You can check the generate pdf `**corr_matrix.pdf` for the correlation matrix, and output log for the accuracy of our predictor