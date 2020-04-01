import vamb
import numpy as np
from sklearn.preprocessing import normalize
import math
import datetime
from collections import defaultdict

def get_tnfs(path):
    with vamb.vambtools.Reader(path, 'rb') as filehandle:
        tnfs, contig_names, lengths_arr = vamb.parsecontigs.read_contigs(filehandle, minlength=4)

    return tnfs, contig_names


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

    for i in range(0, len(bins[0])):
        bins_string += f'{bins[0][i]}\t{bins[1][i]}\n'

    with open('binning_results.tsv', 'w') as output:
        output.write(bins_string)


def get_unique_ids_truth(path):
    with open(path, 'r') as input_file:
        # skips headers
        for _ in range(0,4):
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
