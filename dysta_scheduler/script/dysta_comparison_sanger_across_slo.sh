num_samples=200
samples_per_sec_sweep=( 30 60 120 )
lat_slo_mults=( 1 2 3 4 5 )

for samples_per_sec in "${samples_per_sec_sweep[@]}"; do
    echo "Running with request arrival rate of $samples_per_sec"
    python simulator_across_slo.py \
      --schedule_method fcfs sjf prema planaria sdrm3 dysta \
      --models bart bert gpt2 \
      --csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
      --lat_slo_mult "${lat_slo_mults[@]}" \
      --sample_per_sec $samples_per_sec \
      --num_sample $num_samples
done
