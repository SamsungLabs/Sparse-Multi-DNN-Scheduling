import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def calc_sanger_latency(sparsity, load_balance, seq_len):
    # Q, K, V: [seq_len, QKV_WIDTH]
    # S (original): [seq_len, seq_len]
    # S (after pack & split): [num_subrows, seq_len]

    TH, TW = 64, 64  # PE array: TH x TW
    NUM_PE_PER_ROW = 16
    DATA_TYPE = 2  # 16 bit, 2 Byte
    FREQUENCY = 1e9  # 1G
    REAL_BANDWIDTH = 128  # 128 GB/s

    QKV_WIDTH = 768
    num_subrows = sparsity / load_balance / 0.25 * seq_len  # average number of subrows after pack & split

    LINEAR_GOPS = TH * NUM_PE_PER_ROW * 1 * 2  # pe-size * 1(GHz) * 2(ops/mac) = 2048
    PROJ_GOPS = LINEAR_GOPS / sparsity * load_balance
    
    LAT_linear = seq_len * QKV_WIDTH * QKV_WIDTH * 2 * 3 / 1e9 / LINEAR_GOPS
    LAT_project = seq_len * QKV_WIDTH * QKV_WIDTH * 2 / 1e9 / PROJ_GOPS

    # latency of Q_tile x K_tile
    LAT_Qt_Kt = QKV_WIDTH + TW  # pipeline depth is QKV_WIDTH , TW is the access skew

    # latency of S_tile x V_tile
    LAT_St_Vt = num_subrows + TW  # pipeline depth is QKV_HEIGHT

    # Latency to calculate a THxTW output using QKV
    LAT_TH_TW_output_tile = LAT_Qt_Kt + LAT_St_Vt

    # Latency to calculate a THxSEQUENCE_LENGTH output by changing K and V
    LAT_TH_SEQL_output_tile = LAT_TH_TW_output_tile * (seq_len / TW)

    # to overlap the latency, we need to read TH+TW data per cycle
    # reuse Q in buffer, stream K and V
    required_bandwidth = TW * DATA_TYPE * FREQUENCY / 1e9

    if required_bandwidth < REAL_BANDWIDTH:
        transfer_coeff = 1
    else:
        transfer_coeff = required_bandwidth / REAL_BANDWIDTH

    # final latency to calculate a THxSEQUENCE_LENGTH output
    LAT_final_TH_SEQL_output_tile = LAT_TH_SEQL_output_tile * transfer_coeff

    # final latency to calculate a complete output
    LAT_final = (num_subrows / TH) * LAT_final_TH_SEQL_output_tile 

    return LAT_linear + LAT_final / FREQUENCY + LAT_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", default=None, type=float, required=False)
    parser.add_argument("--load_balance", default=None, type=float, required=False)
    parser.add_argument("--seq_len", default=512, type=int, required=False)
    parser.add_argument("--csv_file", default="load_balance.csv", type=str, required=False, 
                        help="Path to the csv file generated by gen_sparsity_mask.")
    args = parser.parse_args()
    

    if args.sparsity is not None:
        assert args.load_balance is not None
        total_lat = calc_sanger_latency(args.sparsity, args.load_balance, args.seq_len)
        print(f"Sanger Latency: {total_lat * 1000:.3f} ms")
    else:
        assert Path(args.csv_file).exists(), f"{args.csv_file} does not exist."
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(17,11))
        metrics = pd.read_csv(args.csv_file)
        sparsity = metrics['overall-sparsity']
        num_entries = len(sparsity)
        colors = ['pink', 'lightblue', 'lightgreen', 'orange']
        plt_fontsize = 18
        # print(f"Average Sparsity: {sparsity:.3f}")
        for indx, lb_key in enumerate(['50%-skip', '50%-no-skip', '25%-no-skip', '25%-skip']):
            print ("Generating histogram of ", lb_key)
            load_balance = metrics[lb_key]
            batch_latency_dic = {}
            for i in range(num_entries):
                layer_lat = calc_sanger_latency(sparsity[i], load_balance[i], args.seq_len)
                if metrics['batch-indx'][i] not in batch_latency_dic:
                    batch_latency_dic[metrics['batch-indx'][i]] = [layer_lat]
                else:
                    batch_latency_dic[metrics['batch-indx'][i]].append(layer_lat)
            e2e_latency = []
            for k, v in batch_latency_dic.items(): # Accumulate all latency in each key
                e2e_latency.append(sum(batch_latency_dic[k]))
            x = int(indx/2)
            y = int(indx%2)
            ax = axs[x, y]
            ax.hist(e2e_latency, density=True, color=colors[indx], bins=100)  # density=False would make counts
            ax.axvline(np.mean(e2e_latency), color='r', linestyle='dashed', linewidth=2)
            ax.set_ylabel('Probability', fontsize=plt_fontsize)
            ax.set_xlabel("E2E Latency of " + lb_key + " (second)", fontsize=plt_fontsize)
            ax.tick_params(axis='both', labelsize=plt_fontsize-1)
        fig.savefig("/homes/hf17/sanger_e2e_latency.pdf")
            # print(f"Load Balance ({lb_key}): {load_balance:.3f}")
            # print(f"Sanger Latency ({lb_key}): {total_lat * 1000:.3f} ms")


if __name__ == "__main__":
    main()

