# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index

def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)

def transition_matrix_vis(matrix):
    # Heat map
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im)

    ax.set_xticks(range(len((class_to_index))))
    ax.set_yticks(range(len((class_to_index))))
    ax.set_xticklabels(class_to_index.keys())
    ax.set_yticklabels(class_to_index.keys())

    plt.show()

if __name__ == "__main__":
    pass


