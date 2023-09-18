num_samples=1000
samples_per_sec=( 3 4 )
lat_slo_mults=( 10 25 50 75 100 150 )

python simulator.py \
  --schedule_method fcfs sjf prema planaria sdrm3 dysta_oracle dysta \
  --fig_across_slo \
  --csv_lat_dir ./csv_files/eyerissv2/ \
  --lat_slo_mult "${lat_slo_mults[@]}" \
  --sample_per_sec "${samples_per_sec[@]}" \
  --num_sample $num_samples
