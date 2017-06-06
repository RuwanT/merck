import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Global variables
BATCH_SIZE = 64
EPOCH = 200
VAL_FREQ = 5
NET_ARCH = 'merck_net'
MAX_GENERATIONS = 10
PLOT_FEATURE_IMP = False
PLOT_ACCURACY_GEN = True


data_root = '/home/truwan/DATA/merck/preprocessed/'

dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2']


def plot_net_evolve():
    for dataset_name in dataset_names:
        read_header = ['gen_str', 'type', 'run', 'mean', 'std', 'med']
        results_frame = pd.read_csv('./outputs/test_errors_' + dataset_name + '.csv', header=None, names=read_header)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        gen_mean = list()
        for gen in range(0,10):
            gen_mean.append(results_frame.loc[results_frame['type']==str(gen)]['mean'].values)

        ax1.boxplot(gen_mean)

        gen = 0
        read_header = ['gen_str', 'type', 'mean', 'std', 'med', 'nparam']
        gen_results = results_frame.query('type=="stat"')
        gen_results.columns = read_header

        nparam_0 = gen_results.query('gen_str=="Gen_0"')['nparam'].values
        mean_0 = gen_results.query('gen_str=="Gen_0"')['mean'].values
        gen_nparam = list()
        for index, row in gen_results.iterrows():
            ax2.plot(gen, nparam_0 / row['nparam'], 'ro')
            gen_nparam.append(float(nparam_0 / row['nparam']))
            plt.pause(0.1)
            gen += 1

        ax1.set_ylim([0, 2*mean_0])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('RMSE', color='b')
        ax1.tick_params('y', colors='b')
        print gen_nparam, np.percentile(gen_nparam,q=80)
        ax2.set_ylim([0, np.percentile(gen_nparam,q=80)])
        ax2.set_ylabel('Network Efficiency', color='r')
        ax2.tick_params('y', colors='r')

        ax1.axhline(y=mean_0, xmin=0.0, xmax=gen-1, linewidth=1, color='k', linestyle='dashed')

        fig.tight_layout()

        plt.savefig('./outputs/Accuracy_gen_' + dataset_name + '.tiff')
        plt.close()


def print_res_of_gen(gen):
    for dataset_name in dataset_names:
        read_header = ['gen_str', 'type', 'run', 'mean', 'std', 'med']
        results_frame = pd.read_csv('./outputs/test_errors_' + dataset_name + '.csv', header=None, names=read_header)

        read_header = ['gen_str', 'type', 'mean', 'std', 'med', 'nparam']
        gen_results = results_frame.query('type=="stat"')
        gen_results.columns = read_header

        gen_nparam = list()
        for index, row in gen_results.iterrows():
            if str(gen) in  row['gen_str']:
                print dataset_name, row['med'], row['nparam']



if __name__ == "__main__":
    # plot_net_evolve()
    print_res_of_gen(0)