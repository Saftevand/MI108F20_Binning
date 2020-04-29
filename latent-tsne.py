import argparse
import cudf
from cuml import TSNE
import numpy as np
import matplotlib.pyplot as plt

def main():
    args = handle_input_arguments()

    x = np.load(args.latent_file)

    embedded = TSNE(n_components=2).fit_transform(x)
    print(embedded.shape)

    # get the indices where data is 1
    #x, y = np.argwhere(embedded == 1).T
    x = embedded[:, 0]
    y = embedded[:, 1]

    plt.scatter(x, y, s=2)
    plt.savefig('tsne2d.png')





def handle_input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--latent_file", required=True, help="Path to file with latent representations")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
