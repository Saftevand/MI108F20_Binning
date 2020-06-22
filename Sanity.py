import glob
import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import hdbscan
from sklearn.preprocessing import MinMaxScaler


mem_path = 'D:\\datasets\\temp'

pretrain_params = {
        'learning_rate': 0.001,
        'reconst_loss': 'mae',
        'layer_size': 100,
        'num_hidden_layers': 2,
        'embedding_neurons': 50,
        'epochs': [200, 800, 1000, 1000],
        #'epochs': [400],
        #'batch_sizes': [32, 64],
        'batch_sizes': [256, 512, 1024, 4096],
        'activation_fn': 'elu',
        'regularizer': None,
        'initializer': 'he_normal',
        'optimizer': 'Adam',
        'denoise': False,
        'dropout': False,
        'drop_and_denoise_rate': 0.2,
        'BN': False,
        'sparseKLweight': 0.8,
        'sparseKLtarget': 0.1,
        'jacobian_weight': 1e-4,
        'callback_interval': 200
    }
clust_params = {
    'learning_rate': 0.0001,
    'optimizer': 'Adam',
    'loss_weights': [1, 0.05],  # [reconstruction, clustering]
    'jacobian_weight': 1e-4,
    'clustering_weight': 0.2,
    'epochs': 400,
    'reconst_loss': 'mae',
    'clust_loss': 'mae',
    'cuda': True,
    'eps': 0.5,
    'min_samples': 2,
    'min_cluster_size': 6,
    'callback_interval': 50
}

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1, **kwargs):
        self.weight = weight
        self.target = target
        super().__init__(**kwargs)

    def __call__(self, inputs):
        mean_activities = tf.reduce_mean(inputs, axis=0)
        return self.weight * (tf.keras.losses.kld(self.target, mean_activities) + tf.keras.losses.kld(1. - self.target,
                                                                                                      1. - mean_activities))

    def get_config(self):
        return {"weight" : self.weight, "target" : self.target}

def do_clustering(embedding, contig_ids, normalize=True):
    mem = joblib.Memory(location=mem_path)
    min_samples = 10
    min_cluster = 50
    ae_type = "SPARSE"
    dataset = "high"
    normalize_str = f'normalize_{str(normalize)}'
    if normalize:
        scaler = MinMaxScaler()
        embedding = scaler.fit_transform(embedding)
    min_clust_values = [10,20,30,40,50]
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=50, memory=mem).fit(embedding)
    for i in min_clust_values:
        prefix = f'{ae_type}_{dataset}_{normalize_str}_ms{min_samples}_mc{i}'
        clusterer = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=i,
        memory=mem).fit(embedding)
        labels = clusterer.labels_

        bins = get_assignments(labels, contig_ids)
        write_bins_to_file(bins=bins, prefix=prefix)




def sensitivity(embedding_matrix, loss, x_train, decoder, tnf_feature_size=136, abundance_feature_size=2):
    #model = tf.keras.models.load_model
    dataset_variable = tf.Variable(embedding_matrix)
    with tf.GradientTape() as tape:
        reconstructed = decoder(dataset_variable, training=True)
        # loss = tf.reduce_mean(tf.keras.losses.MAE(dataset_variable, reconstructed))
        loss = loss(x_train, reconstructed)
    gradients = tape.gradient(loss, dataset_variable)
    gradients = tf.reduce_mean(tf.abs(gradients), 0)
    number_of_features = embedding_matrix.shape[1]


    #tnf_feature = [f'TNF\n{e}' for e in range(1, tnf_feature_size + 1)]
    #abundance_feature = [f'ABD\n{e}' for e in range(1, number_of_features - (tnf_feature_size - 1))]
    shape = int(np.ceil(np.sqrt(number_of_features)))

    mask = np.zeros(shape * shape, dtype=bool)
    for index in range(number_of_features, shape * shape):
        mask[index] = True
    mask = mask.reshape(shape, shape)

    gradient_list = np.asarray(gradients.numpy()).tolist()
    sum_gradients_percentage =100 / np.sum(gradient_list)
    dimensions = [f'D{e}\n{(sum_gradients_percentage*gradient_list[e-1]):.2f}%' for e in range(1, number_of_features + 1)]
    labels = dimensions
    labels += list(range(0, (shape * shape) - number_of_features))
    labels = np.asarray(labels).reshape(shape, shape)

    max_gradient = max(gradient_list)
    min_gradient = min(gradient_list)
    for _ in range((shape * shape) - number_of_features):
        gradient_list.append(0.0)
    np_gradients = np.asarray(gradient_list).reshape(shape, shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(np_gradients, ax=ax, vmin=min_gradient, vmax=max_gradient, annot=labels, fmt='', mask=mask)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # plt.show()
    plt.savefig('Evaluate_embedding\\sensitivity.png')
    return gradients


def loss(y_pred, y_true):
    s = 5
    y_true_tnfs = y_true[:, :-s]
    y_true_abd = y_true[:, -s:]

    y_pred_tnfs = y_pred[:, :-s]
    y_pred_abd = y_pred[:, -s:]

    # TNF error
    # tnf_diff = tf.abs(y_true_tnfs - y_pred_tnfs)

    # tnf_err = tf.reduce_mean(tf.reduce_sum(tnf_diff, axis=1))
    tnf_err = tf.losses.MAE(y_pred_tnfs, y_true_tnfs)

    # ABUNDANCE error
    # abd_diff = tf.abs(y_true_abd - y_pred_abd)
    # abd_err = tf.reduce_mean(tf.reduce_sum(abd_diff, axis=1))
    abd_err = tf.losses.MAE(y_pred_abd, y_true_abd)
    # total_abd_err = tf.reduce_sum(-(tf.math.log(y_pred_abd + 1e-9)) * y_true_abd, axis=1)

    # ratio = np.ceil(num_tnfs / s)
    # loss = tnf_err/ s + abd_err
    # loss = (tnf_err / (s + num_tnfs)) + abd_err
    loss_value = tnf_err/(s*40) + abd_err
    return loss_value


def get_assignments(labels, contig_ids, include_outliers=False):

    if include_outliers is False:
        outlier_mask = (labels != -1)
    else:
        outlier_mask = np.ones(len(labels), dtype=bool)

    try:
        bins = np.vstack([contig_ids[outlier_mask], labels[outlier_mask]])
    except:
        print('Could not combine contig ids with bin assignment')
        print('\nContig IDs:', contig_ids, '\nBins:', labels)
    return bins


def write_bins_to_file(bins, prefix):

    output_string = '@Version:0.9.1\n@SampleID:gsa\n\n@@SEQUENCEID\tBINID\tLENGTH\n'

    for i in range(0, len(bins[0])):

        output_string += f'{bins[0][i]}\t{bins[1][i]}\n'

    with open(f'Embedding_cluster_tests\\{prefix}_binning_results.tsv', 'w') as output:
        output.write(output_string)



#embedding_stacked_high = np.load('Evaluate_embedding\\stacked_high\\embedding_STACKED.npy')
#embedding_stacked_airways = np.load('Evaluate_embedding\\stacked_airways\\embedding_STACKED.npy')
embedding_sparse_high = np.load('Evaluate_embedding\\sparse_high\\embedding_SPARSE.npy')
#embedding_sparse_airways = np.load('Evaluate_embedding\\sparse_airways\\embedding_SPARSE.npy')

contig_ids_high = np.load('D:\\datasets\\cami_high\\contig_ids_high.npy')
#contig_ids_airways = np.load('D:\\datasets\\cami_airways\\contig_ids_high.npy')

labels = do_clustering(embedding=embedding_sparse_high, contig_ids=contig_ids_high)
#bins = get_assignments(labels, contig_ids_high, include_outliers=False)
#write_bins_to_file(bins)




x_train_airways = np.load('Evaluate_embedding\\cami_airways_x_train.npy')
x_train_high = np.load('Evaluate_embedding\\cami_high_x_train.npy')

'''
embedding_stacked_high_mean = np.mean(embedding_stacked_high,axis=0)
embedding_stacked_airways_mean = np.mean(embedding_stacked_airways, axis=0)
embedding_sparse_high_mean = np.mean(embedding_sparse_high,axis=0)
embedding_sparse_airways_mean = np.mean(embedding_sparse_airways,axis=0)

embedding_stacked_high_std = np.std(embedding_stacked_high,axis=0)
embedding_stacked_airways_std = np.std(embedding_stacked_airways, axis=0)
embedding_sparse_high_std = np.std(embedding_sparse_high,axis=0)
embedding_sparse_airways_std = np.std(embedding_sparse_airways,axis=0)

embedding_sparse_odd_high = np.load('Evaluate_embedding\\sparse_loss_weight2_kl_target0.05_weight0.1\\high\\embedding_SPARSE.npy')
embedding_sparse_odd_airways = np.load('Evaluate_embedding\\sparse_loss_weight2_kl_target0.05_weight0.1\\airways\\embedding_SPARSE.npy')
embedding_sparse_odd_high_mean = np.mean(embedding_sparse_odd_high,axis=0)
embedding_sparse_odd_airways_mean = np.mean(embedding_sparse_odd_airways,axis=0)
embedding_sparse_odd_airways_std = np.std(embedding_sparse_odd_airways,axis=0)
embedding_sparse_odd_high_std = np.std(embedding_sparse_odd_high,axis=0)
results_mean = np.mean(np.vstack([embedding_stacked_high_std, embedding_stacked_airways_std, embedding_sparse_high_std, embedding_sparse_airways_std, embedding_sparse_odd_high_std, embedding_sparse_odd_airways_std]), axis=1)

matrix_of_means = np.transpose(np.vstack([embedding_stacked_high_std, embedding_stacked_airways_std, embedding_sparse_high_std, embedding_sparse_airways_std, embedding_sparse_odd_high_std, embedding_sparse_odd_airways_std]))

results = np.transpose(np.abs(matrix_of_means - results_mean))

model = tf.keras.models.load_model('Evaluate_embedding\\stacked_high\\autoencoder_STACKED', compile=False)
#tf.keras.regularizers.KLDivergenceRegularizer = KLDivergenceRegularizer
#model = tf.keras.models.load_model('Evaluate_embedding\\sparse_high\\autoencoder_SPARSE', compile=False, custom_objects={
#            "KLDivergenceRegularizer": KLDivergenceRegularizer})

model.compile(loss=loss)

input = tf.keras.layers.Input(shape=(32,), name="input")

temp = model.get_layer('decoding_layer_0')(input)

for layer in model.layers[-4:]:
    temp = layer(temp)

decoder = tf.keras.Model(input, temp, name="Decoder")
print(decoder.summary())

sensitivity(embedding_matrix=embedding_stacked_high,loss=loss, x_train=x_train_high,decoder=decoder, abundance_feature_size=None, tnf_feature_size=None)
'''




test = np.array([[1,2,3],[1,2,3],[1,2,3]])
tensor = tf.convert_to_tensor(test)
split = test[:,1:2]


config = {}
config['pretrain_params'] = pretrain_params
config['clust_params'] = clust_params
print('test')