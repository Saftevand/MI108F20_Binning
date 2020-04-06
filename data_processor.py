import vamb
import numpy as np
from sklearn.preprocessing import normalize
import math
import datetime
from collections import defaultdict
import cuml
import cudf
import pandas as pd
import binner
# import matplotlib.pyplot as plt

def get_tnfs(path):
    with vamb.vambtools.Reader(path, 'rb') as filehandle:
        tnfs, contig_names, lengths_arr = vamb.parsecontigs.read_contigs(filehandle, minlength=4)

    return tnfs, contig_names

def plot_cluster_optimizing_loss(list_of_loss):
    plt.plot(list_of_loss)
    plt.title('Clustering loss')
    plt.ylabel('Loss value')
    plt.xlabel('No. batch?? TODO')
    plt.legend(loc="upper left")
    return plt.gcf()  # Get Current Figure


def plot_history_obj(history_obj, metric):
    plt.plot(history_obj.history[metric])
    plt.title(f'Finetuning {metric}.')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    return plt.gcf()  # Get Current Figure


def plot_layerwise(history_list):
    list_of_plots = []
    counter = 1
    for hist in history_list:
        plt.plot(hist.history['loss'])
        plt.title(f'layer {counter} loss.')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        list_of_plots.append(plt.gcf())
    return list_of_plots


def write_training_plots(binner_instance, out_dir):

    if binner_instance is binner.DEC_Binner_Xifeng:
        # plots pretraining loss
        plot1 = plot_history_obj(binner_instance.full_AE_train_history, 'loss')
        # plots cluster loss
        plot2 = plot_cluster_optimizing_loss(binner_instance.cluster_loss_list)

        plot1.savefig(f'{out_dir}/pretrain_loss')
        plot2.savefig(f'{out_dir}/cluster_fit_loss')
        plt.close(plot1)
        plt.close(plot2)

    elif binner_instance is binner.Greedy_pretraining_DEC:
        # plots layerwise training
        plot_list = plot_layerwise(binner_instance.layers_history)
        # plots pretraining loss
        plot1 = plot_history_obj(binner_instance.full_AE_train_history, 'loss')
        # plots cluster loss
        plot2 = plot_cluster_optimizing_loss(binner_instance.cluster_loss_list)

        layernr = 1
        for plot in plot_list:
            plot.savefig(f'{out_dir}/layer_{layernr}_loss')

        plot1.savefig(f'{out_dir}/pretrain_loss')
        plot2.savefig(f'{out_dir}/cluster_fit_loss')
        plt.close()

    elif binner_instance is binner.Sequential_Binner:
        # plots AE training loss
        plot1 = plot_history_obj(binner_instance.full_AE_train_history, 'loss')
        plot1.savefig(f'{out_dir}/pretrain_loss')
        plt.close(plot1)



def get_depth(paths):
    return vamb.parsebam.read_bamfiles(paths)


def get_featurematrix(args):

    if args.bam:
        depth = get_depth(args.bam)
        depth = normalize(depth, axis=1, norm='l1')

    else:
        depth = np.load(args.loaddepth)
        depth = normalize(depth, axis=1, norm='l1')

    if args.read:
        tnfs, contig_ids = get_tnfs(args.read)
    else:
        tnfs = np.load(args.loadtnfs)
        contig_ids = np.load(args.loadcontigids)

    if args.savepathtnfs:
        np.save(args.savepathtnfs + "tnfs",tnfs)
        np.save(args.savepathcontigids + "contigids",contig_ids)

    if args.savepathdepth:
        np.save(args.savepathdepth +"depth", depth)


    feature_matrix = np.hstack([tnfs, depth])
    return feature_matrix, contig_ids


def get_train_and_validation_data(feature_matrix, split_value=0.8):
    # TODO måske skal ham her være lidt bedre. Det er farligt at tage de første x % hver gang
    split_length = math.floor(len(feature_matrix) * split_value)
    train = feature_matrix[:split_length]
    validate = feature_matrix[split_length + 1:]
    return train, validate


def write_bins_to_file(bins):
    bins_string = "@@SEQUENCEID\tBINID\tLENGTH\n"
    print(bins)

    for i in range(0, len(bins[0])):
        bins_string += f'{bins[0][i]}\t{bins[1][i]}\n'

    with open('binning_results.tsv', 'w') as output:
        output.write(bins_string)


def get_train_and_test_data(data, split_value=0.8):
    x_train, x_test, y_train, y_test = cuml.model_selection.train_test_split(data, 'y', train_size=split_value)  # tror y her skal være den ene dimension i datasettet
    return x_train, x_test, y_train, y_test

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
      pdf[str(c)] = df[column]
    return pdf
