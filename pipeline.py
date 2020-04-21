import cluster
import numpy as np
import tensorflow as tf
import vamb_clust as vamb_clust
import torch as _torch


def cluster_ite(clusters):
    clusternumber = 0
    ncontigs = 0
    min_size = 1
    for clustername, contigs in clusters:
        if len(contigs) < min_size:
            continue

        if True:
            clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t')

        clusternumber += 1
        ncontigs += len(contigs)

        #if clusternumber == max_clusters:
        #    break

    return clusternumber, ncontigs


#normal = np.random.normal(2,1,100000)
#contig_names = np.arange(10000)
#normal = normal.reshape((10000,10))
#normal_tf = tf.convert_to_tensor(normal)
#normal_torch = _torch.tensor(normal)


#normal_tf = cluster.normalize(normal_tf)
#dist_tf = cluster.calc_distances(normal_tf, 0)
#normal_torch = snippet.normalize(normal_torch)
#dist_torch = snippet.calc_distances(normal_torch, 0)

latent_representation = np.load('latent_representation.npy')
latent_representation_tf = tf.Variable(latent_representation)
contig_names = np.load('contig_ids_high.npy')

#it = vamb_clust.cluster(latent_representation)
it = cluster.cluster(latent_representation_tf)
renamed = ((str(i+1), c.as_tuple(contig_names)[1]) for (i, c) in enumerate(it))
cluster_ite(renamed)
print("meh")
