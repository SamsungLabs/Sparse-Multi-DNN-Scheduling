# Draw correlation of sparsity, Calcuate the accurcay of sparsity predictor
python dysta_lat_pred.py --models bert gpt2 \
--csv_lat_files  ./csv_files/sanger/load_balance_bert_squad_384.csv ./csv_files/sanger/load_balance_gpt2_glue_128.csv \
--draw_corr_matrix
