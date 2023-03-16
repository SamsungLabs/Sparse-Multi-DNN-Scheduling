# Evaluate arrival rate with 120, num of sample 200
# The num of sample is set according to arrival rate, but feel free to change it.
python simulator.py --schedule_method fcfs prema prema_sparse sjf sjf_sparse dysta \
--models bart bert gpt2 --csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
--lat_slo_mult 1 2 3 4 5 --sample_per_sec 120 --num_sample 200

# Evaluate arrival rate with 60, num of sample 100
python simulator.py --schedule_method fcfs prema prema_sparse sjf sjf_sparse dysta \
--models bart bert gpt2 --csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
--lat_slo_mult 1 2 3 4 5 --sample_per_sec 60 --num_sample 100

# Evaluate arrival rate with 30, num of sample 50
python simulator.py --schedule_method fcfs prema prema_sparse sjf sjf_sparse dysta \
--models bart bert gpt2 --csv_lat_files ./csv_files/sanger/load_balance_bart_glue_128.csv  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
--lat_slo_mult 1 2 3 4 5 --sample_per_sec 30 --num_sample 50
