num_samples=1000
samples_per_sec=( 30 40 )
lat_slo_mults=( 10 )

python simulator.py \
  --schedule_method fcfs sjf prema planaria sdrm3 dysta \
  --fig_tradeoff_analyse \
  --csv_lat_dir ./csv_files/sanger/ \
  --lat_slo_mult "${lat_slo_mults[@]}" \
  --sample_per_sec "${samples_per_sec[@]}" \
  --num_sample $num_samples
