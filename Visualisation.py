# Basic graphs and so on
import matplotlib.pyplot as plt
import json


def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file

        return json.load(openfile)

def transition_matrix_vis():
    # Heat map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(transitions_matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im)
    plt.show()


transitions_matrix = open_dict('transition_matrix.json')
transition_matrix_vis()

