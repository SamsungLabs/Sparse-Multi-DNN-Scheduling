num_samples=1000
samples_per_sec_sweep=( 30 60 90 120 )
samples_per_sec_sweep=( 10 20 30 40 )
lat_slo_mults=( 10 )

for lat_slo_mult in "${lat_slo_mults[@]}"; do
    echo "Running with request arrival rate sweep for SLO $lat_slo_mult"
    python simulator_across_arrival_rates.py \
      --schedule_method fcfs sjf prema planaria sdrm3 dysta \
      --models bart bert gpt2 \
      --csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
      --lat_slo_mult $lat_slo_mult \
      --sample_per_sec "${samples_per_sec_sweep[@]}" \
      --num_sample $num_samples
done
