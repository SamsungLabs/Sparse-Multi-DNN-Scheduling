# Run on three models
python simulator.py --schedule_method fcfs prema dysta \
--models bart bert gpt2 \
--csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
--lat_slo_mult 1 2 3 4 5 --sample_per_sec 30 --num_sample 200