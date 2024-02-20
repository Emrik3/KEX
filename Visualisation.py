# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index
import numpy as np


def transition_matrix_vis(matrix):
    """Generates heat map of transition matrix """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.flipud(matrix), cmap='viridis', interpolation='nearest')
    fig.colorbar(im)
    mat_size = max(class_to_index.values()) + 1
    ax.set_xticks(range(mat_size))
    ax.set_yticks(range(mat_size-1,-1,-1))

    keys_to_include = list(class_to_index.keys())[0:mat_size]
    print(class_to_index)
    ax.set_xticklabels(keys_to_include)
    ax.set_yticklabels(keys_to_include)
    plt.show()

if __name__ == "__main__":
    pass


