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
from sklearn.model_selection import train_test_split
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

    feature_matrix, x_train, x_valid = preprocess_data(tnfs, depth)
    #feature_matrix = np.hstack([tnfs, depth])
    #np.save("full_feature_matrix", feature_matrix)

    return feature_matrix, contig_ids, x_train, x_valid

def preprocess_data(tnfs, depths):

    tnf_shape = tnfs.shape[1]
    depth_shape = depths.shape[1]
    number_of_features = tnf_shape + depth_shape
    tnf_weight = tnf_shape / number_of_features
    depth_weight = depth_shape / number_of_features
    weighted_tnfs = tnfs * tnf_weight
    weighted_depths = depths * depth_weight

    # feature_matrix = np.hstack([weighted_tnfs, weighted_depths])
    feature_matrix = np.hstack([tnfs, depths])
    x_train, x_valid = train_test_split(feature_matrix, test_size=0.2, shuffle=True)
    training_mean = np.mean(x_train, axis=0)
    training_std = np.std(feature_matrix, axis=0)

    x_train -= training_mean
    x_train /= training_std

    x_valid -= training_mean
    x_valid /= training_std
    feature_matrix -= training_mean
    feature_matrix /= training_std
    return feature_matrix, x_train, x_valid

def get_train_and_validation_data(feature_matrix, split_value=0.8):
    # TODO måske skal ham her være lidt bedre. Det er farligt at tage de første x % hver gang
    split_length = math.floor(len(feature_matrix) * split_value)
    train = feature_matrix[:split_length]
    validate = feature_matrix[split_length + 1:]
    return train, validate


def write_bins_to_file(bins):
    bins_string = '@Version:0.9.1\n@SampleID:gsa\n\n@@SEQUENCEID\tBINID\tLENGTH\n'

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

def get_cami_data_truth(path):
    with open(path, 'r') as input_file:
        # skips headers
        for _ in range(0, 4):
            next(input_file)

        ids = []
        contig_ids = []
        binid_to_int = defaultdict()
        contigid_to_binid = defaultdict()

        new_id = 0
        for line in input_file:
            line_elems = line.split('\t')
            bin_id = line_elems[1]
            cont_name = line_elems[0]
            contig_ids.append(line_elems[0])

            if bin_id in binid_to_int:
                id_int = binid_to_int[bin_id]

                # used for mapping contig id to bin id
                contigid_to_binid[cont_name] = id_int
            else:
                # new bin id
                binid_to_int[bin_id] = new_id
                id_int = new_id

                # used for mapping contig id to bin id
                contigid_to_binid[cont_name] = id_int

                new_id += 1

            ids.append(id_int)

    return ids, contig_ids, contigid_to_binid


def sort_bins_follow_input(contig_ids: [int], contig_to_bin_id: defaultdict):
    var = []
    for i in contig_ids:
        var.append(contig_to_bin_id[i])
    return var