num_samples=1000
samples_per_sec=( 30 )
lat_slo_mults=( 10 )

# SJF
# python simulator.py \
#   --schedule_method prema prema_sparse \
#   --fig_sparse_effect \
#   --csv_lat_dir ./csv_files/sanger/ \
#   --lat_slo_mult "${lat_slo_mults[@]}" \
#   --sample_per_sec "${samples_per_sec[@]}" \
#   --num_sample $num_samples

# SDRM3
# python simulator.py \
#   --schedule_method sdrm3 sdrm3_sparse\
#   --fig_sparse_effect \
#   --csv_lat_dir ./csv_files/sanger/ \
#   --lat_slo_mult "${lat_slo_mults[@]}" \
#   --sample_per_sec "${samples_per_sec[@]}" \
#   --num_sample $num_samples

# Dysta
python simulator.py \
  --schedule_method prema dysta_naive dysta \
  --fig_sparse_effect \
  --csv_lat_dir ./csv_files/sanger/ \
  --lat_slo_mult "${lat_slo_mults[@]}" \
  --sample_per_sec "${samples_per_sec[@]}" \
  --num_sample $num_samples

