#import vamb
import numpy as np
from sklearn.preprocessing import normalize
import os
import math
import datetime
from collections import defaultdict
#import cuml
#import cudf
#import pandas as pd
#import binner
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_tnfs(path):
    with vamb.vambtools.Reader(path, 'rb') as filehandle:
        tnfs, contig_names, lengths_arr = vamb.parsecontigs.read_contigs(filehandle, minlength=4)

    return tnfs, contig_names

def get_depth(paths):
    return vamb.parsebam.read_bamfiles(paths)


def get_featurematrix(args, labels=None):

    if args.bam:
        depth = get_depth(args.bam)

    else:
        depth = np.load(args.loaddepth)

    if args.read:
        tnfs, contig_ids = get_tnfs(args.read)
    else:
        tnfs = np.load(args.loadtnfs)
        contig_ids = np.load(args.loadcontigids)

    '''
    if args.savepathtnfs:
        np.save(args.savepathtnfs + "tnfs",tnfs)
        np.save(args.savepathcontigids + "contigids",contig_ids)

    if args.savepathdepth:
        np.save(args.savepathdepth +"depth", depth)
    '''

    feature_matrix, x_train, x_valid, train_labels, validation_labels, samples = preprocess_data(tnfs, depth)

    return feature_matrix, contig_ids, x_train, x_valid, train_labels, validation_labels, samples

def preprocess_data(tnfs, depths, labels=None, use_validation_data=False):
    tnf_shape = tnfs.shape[1]
    samples = depths.shape[1]

    depths = normalize(depths, axis=1, norm='l1')
    tnfs = normalize(tnfs, axis=1, norm='l1')
    feature_matrix = np.hstack([tnfs, depths])

    tnfs_mean = np.mean(tnfs, axis=0)
    tnfs -= tnfs_mean
    tnfs_std = np.std(tnfs, axis=0)
    tnfs /= tnfs_std

    x_train = np.hstack([tnfs, depths])
    train_labels = labels
    x_valid = []
    validation_labels = []

    return feature_matrix, x_train, x_valid, train_labels, validation_labels



def write_bins_to_file(bins, output_dir=''):

    output_string = '@Version:0.9.1\n@SampleID:gsa\n\n@@SEQUENCEID\tBINID\tLENGTH\n'

    for i in range(0, len(bins[0])):

        output_string += f'{bins[0][i]}\t{bins[1][i]}\n'

    with open((os.path.join(output_dir, 'binning_results.tsv')), 'w') as output:
        output.write(output_string)


def get_cami_data_truth(path):
    with open(path, 'r') as input_file:
        # skips headers
        for _ in range(0, 4):
            next(input_file)

        ids = []
        contig_ids = []
        binid_to_int = defaultdict()
        contigid_to_binid = defaultdict()
        bin_id_to_contig_names = defaultdict(list)

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

    for contig_name, binid in contigid_to_binid.items():
        bin_id_to_contig_names[binid].append(contig_name)
    d = {int(k.split("|C")[1]): int(v) for k, v in contigid_to_binid.items()}
    contig_id_binid_sorted = dict(sorted(d.items()))
    return ids, contig_ids, contigid_to_binid, contig_id_binid_sorted


def load_data_local(dataset_path):

    dataset_path = dataset_path
    tnfs = np.load(os.path.join(dataset_path, "tnfs_high.npy"))
    contig_ids = np.load(os.path.join(dataset_path, "contig_ids_high.npy"))
    depth = np.load(os.path.join(dataset_path, "depths_high.npy"))
    return tnfs, contig_ids, depth


def sort_bins_follow_input(contig_ids: [int], contig_to_bin_id: defaultdict):
    var = []
    for i in contig_ids:
        var.append(contig_to_bin_id[i])
    return var