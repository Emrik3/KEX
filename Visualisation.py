# Basic graphs and so on
import matplotlib.pyplot as plt
import json

def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)

def transition_matrix_vis(matrix):
    # Heat map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im)
    plt.show()

if __name__ == "__main__":
    pass


