## Instructions for Generating Figures & Tables

We provide the script to reproduce all the figures/tables presented in our paper.

### Figure-2
To visualize the impact of dynamic sparsity on language models, run:
```
cd dysta_scheduler/
bash script/latency_profile_bert_motivation.sh 
``` 

### Figure-3 & Table-2
To visualize the sparsity ratio of activations in CNNs, run:
```
cd dataset_sparsity/
bash draw_sparsity_motivation.sh
``` 

### Figure-4
To visualize the impact of different sparsity patterns on CNNs, run:

```
cd dataset_sparsity/
bash sparsity_pattern_analysis.sh 
```
Note: In `dataset_sparsity/sparsity_pattern_analysis.sh`, replace the `--dataset_root` and `--figs_path` to your own path to dataset and figure folders.


### Figure-9 & Table-4
Visualize the correlation of sparsity ratio of different layers, run:
```
cd dysta_scheduler/
bash script/dysta_lat_pred_draw_correlation.sh 
```

### Figure-12
Evaluate the SLO violation rate and ANTT trade-off of different scheduling approaches:
```
cd dysta_scheduler/
# Attention Results
bash script/attnn/dysta_comparison_sanger_tradeoff_analysis.sh
# CNN Results
bash script/cnn/dysta_comparison_eyerissv2_tradeoff_analysis.sh
```

### Figure-13
Evaluate the optimization breakdown of our proposed Sparse-Dysta scheduler:
```
cd dysta_scheduler/
# Attention Results
bash script/attnn/effect_sparsity_sanger.sh
# CNN Results
bash script/cnn/effect_sparsity_eyerissv2.sh 
```

### Table-5 & Figure-14
Stress test on different SLO latency multipliers:
```
cd dysta_scheduler/
# Attention Results
bash script/attnn/dysta_comparison_sanger_across_slo.sh
# CNN Results
bash script/cnn/dysta_comparison_eyerissv2_across_slo.sh
```

### Figure-15
Stress test on different arrival rates:
```
cd dysta_scheduler/
# Attention Results
bash script/attnn/dysta_comparison_sanger_across_arrival_rates.sh
# CNN Results
bash script/cnn/dysta_comparison_eyerissv2_across_arrival_rates.sh
```

### Figure-16 & Table-6

Download the Vivado Place-&-Route reports from [here](https://drive.google.com/drive/folders/1OcTIqF1nYl-7CEH0_VI6dFULcXZe2PWm?usp=sharing) and save it under `hw_design/vivado_rpt/`. Run:
```
cd hw_design/
bash draw_fig_hw_opt.sh
```
To quickly check the resource consumption of Table-6, you can download the Vivado reports from [here](https://drive.google.com/drive/folders/1OcTIqF1nYl-7CEH0_VI6dFULcXZe2PWm?usp=sharing). You can also use Vivado projects under the path of `/workspace/vivado_projects` in the Docker image to re-run synthesis and place&route.
