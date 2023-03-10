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