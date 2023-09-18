num_samples=1000
samples_per_sec=( 3 )
lat_slo_mults=( 10 )

python simulator.py \
  --schedule_method prema dysta_naive dysta \
  --fig_sparse_effect \
  --csv_lat_dir ./csv_files/eyerissv2/ \
  --lat_slo_mult "${lat_slo_mults[@]}" \
  --sample_per_sec "${samples_per_sec[@]}" \
  --num_sample $num_samples
