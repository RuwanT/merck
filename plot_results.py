import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from nutsflow import *
from nutsml import *

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


# dataset_names = ['CB1', 'LOGD']


def plot_net_evolve():
    for dataset_name in dataset_names:
        read_header = ['gen_str', 'type', 'run', 'mean', 'std', 'med']
        results_frame = pd.read_csv(
            "C:\\GIT_codes\\backup_models\\outputs_train_evo_net\\test_errors_" + dataset_name + '.csv', header=None,
            names=read_header)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        gen_mean = list()
        for gen in range(0, 10):
            gen_mean.append(results_frame.loc[results_frame['type'] == str(gen)]['mean'].values)

        ax1.boxplot(gen_mean)

        gen = 0
        read_header = ['gen_str', 'type', 'mean', 'std', 'med', 'nparam']
        gen_results = results_frame.query('type=="stat"')
        gen_results.columns = read_header

        nparam_0 = gen_results.query('gen_str=="Gen_0"')['nparam'].values
        mean_0 = gen_results.query('gen_str=="Gen_0"')['mean'].values
        gen_nparam = list()
        for index, row in gen_results.iterrows():
            gen_nparam.append(float(nparam_0 / row['nparam']))
            # plt.pause(0.1)
            gen += 1

        ax2.plot(range(1, gen + 1), gen_nparam, 'ro-')
        plt.pause(0.1)

        ax1.set_ylim([0, 2 * mean_0])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('RMSE', color='b')
        ax1.tick_params('y', colors='b')
        print gen_nparam, np.percentile(gen_nparam, q=60)
        ax2.set_ylim([0, np.percentile(gen_nparam, q=60)])
        ax2.set_ylabel('Network Efficiency', color='r')
        ax2.tick_params('y', colors='r')

        ax1.axhline(y=mean_0, xmin=0.0, xmax=gen - 1, linewidth=1, color='k', linestyle='dashed')

        fig.tight_layout()

        plt.savefig('C:\\GIT_codes\\backup_models\\outputs_train_evo_net\\Accuracy_gen_' + dataset_name + '.tiff')
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
            if str(gen) in row['gen_str']:
                print dataset_name, row['med'], row['nparam']


def selected_feature_overalp():
    # Use average correlation to compare the features selected in different runs of the neural network
    GEN = 10
    NNRUN = 5
    np.set_printoptions(precision=3)
    res = list()
    for dataset_name in dataset_names:

        train_file = data_root + dataset_name + '_training_disguised.csv'
        assert os.path.isfile(train_file)
        data_test = ReadPandas(train_file, dropnan=True)
        data_frm = data_test.dataframe.ix[:, 2:]
        data_corr = data_frm.corr().values

        read_header = ['nrun', 'gen', 'test_RMSE', 'test_R2', 'val_RMSE', 'num_features']
        results_frame = pd.read_csv(
            "./outputs/feature_selection_bm_" + dataset_name + '.csv', header=None, names=read_header)
        temp = dict()
        for nrun in range(0, NNRUN):
            feature_npy_name = './outputs/featureSelect_bm_' + dataset_name + '_' + str(nrun) + '_' + str(GEN) + '.npy'
            if os.path.isfile(feature_npy_name):
                temp[nrun] = list(np.nonzero(np.load(feature_npy_name))[0])

        out = np.zeros((NNRUN, NNRUN), dtype=float)
        for nrun_i, sel_features_i in temp.iteritems():
            for nrun_j, sel_features_j in temp.iteritems():
                if nrun_j <= nrun_i:
                    continue

                out[nrun_i, nrun_j] = np.mean(np.max((data_corr[sel_features_i, :])[:, sel_features_j], axis=1)) * 100
                print dataset_name, nrun_j, nrun_i, out[nrun_i, nrun_j], \
                    results_frame.loc[results_frame['nrun'] == nrun_i]['test_RMSE'].values[GEN], \
                    results_frame.loc[results_frame['nrun'] == nrun_j]['test_RMSE'].values[GEN], len(
                    sel_features_i), len(sel_features_j)

                res.append([dataset_name, nrun_j, nrun_i, out[nrun_i, nrun_j],
                            results_frame.loc[results_frame['nrun'] == nrun_i]['test_RMSE'].values[GEN],
                            results_frame.loc[results_frame['nrun'] == nrun_j]['test_RMSE'].values[GEN],
                            len(sel_features_i), len(sel_features_j)])

        print out

    writer = WriteCSV('./outputs/feature_overlap.csv')
    res >> writer


def plot_feature_selection():
    for dataset_name in dataset_names:

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        read_header = ['nrun', 'gen', 'test_RMSE', 'test_R2', 'val_RMSE', 'num_features']
        results_frame = pd.read_csv(
            "./outputs/feature_selection_bm_" + dataset_name + '.csv', header=None, names=read_header)

        print 'Plotting dataset ' + dataset_name

        run_rmse = list()
        nfeatures = list()
        for nrun in range(0, 5):
            run_rmse.append(results_frame.loc[results_frame['nrun'] == nrun]['test_RMSE'].values)
            nfeatures.append(results_frame.loc[results_frame['nrun'] == nrun]['num_features'].values)

        run_rmse = (np.asarray(run_rmse))

        ax1.boxplot(run_rmse)
        ax2.plot(range(1, 10 + 2), np.mean(np.asarray(nfeatures), axis=0, keepdims=False), 'ro-')
        ax2.text(10.5, np.asarray(nfeatures).flatten()[-1] + 10, '$n_f$ = ' + str(np.asarray(nfeatures).flatten()[-1]), verticalalignment='bottom', horizontalalignment='center', fontsize=15)

        ax1.set_ylim([0, np.max(run_rmse) * 1.1])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('RMSE', color='b')
        ax1.tick_params('y', colors='b')
        ax2.set_ylim([0, np.max(nfeatures) * 1.1])
        ax2.set_ylabel('Number of Features', color='r')
        ax2.tick_params('y', colors='r')
        ax1.axhline(y=np.mean(results_frame.loc[results_frame['gen'] == 0]['test_RMSE'].values), xmin=0.0, xmax=11 - 1, linewidth=1, color='k', linestyle='dashed')
        plt.pause(.1)

        fig.tight_layout()

        plt.savefig('./outputs/feature_select_' + dataset_name + '.tiff')
        plt.close()


def plot_corr():
    import scipy
    import pylab
    import scipy.cluster.hierarchy as sch

    dataset_name = 'CB1'
    test_file = data_root + dataset_name + '_test_disguised.csv'
    data_test = ReadPandas(test_file, dropnan=True)
    data_frm = data_test.dataframe.ix[:, 2:]
    # data_frm = np.abs(data_frm.corr().values)

    import seaborn as sns
    sns.set()

    g = sns.clustermap(data_frm.corr())
    g.savefig("a.png")
    # # Compute and plot dendrogram.
    # fig = pylab.figure()
    # axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    # Y = sch.linkage(data_frm, method='centroid')
    # Z = sch.dendrogram(Y, orientation='right')
    # axdendro.set_xticks([])
    # axdendro.set_yticks([])
    #
    # # Plot distance matrix.
    # axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    # index = Z['leaves']
    # D = data_frm[index, :]
    # D = D[:, index]
    # im = axmatrix.matshow(D, aspect='auto', origin='lower')
    # axmatrix.set_xticks([])
    # axmatrix.set_yticks([])
    #
    # # Plot colorbar.
    # axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    # pylab.colorbar(im, cax=axcolor)
    #
    # # Display and save figure.
    # fig.show()
    # fig.savefig('a.png')


    # plt.imshow(data_frm, interpolation="nearest", cmap='jet')
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    # plot_net_evolve()
    # print_res_of_gen(0)
    # selected_features()
    plot_feature_selection()
    # selected_feature_overalp()
    # plot_corr()
