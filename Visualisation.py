# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index
import numpy as np
from collections import Counter


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

def organize_and_plot(res):
    wrong_predicted_class = [] # When the model predicted wrong it predicted these wc
    wrong_actual_class = [] # When the model predicted wrong it should have predicted these wc
    corr_actual_class = [] # When the model predicted right it predicted these wc

    for elem in res:
        wrong_predicted_class.append(elem[0])
        wrong_actual_class.append(elem[1])
        corr_actual_class.append(elem[2])
    wrong_predicted_class = sum(wrong_predicted_class,[])
    wrong_actual_class = sum(wrong_actual_class,[])
    corr_actual_class = sum(corr_actual_class,[])

    # new list with all the wc that appeared in the test text
    all_actual_classes = wrong_actual_class + corr_actual_class

    # Counts how many times each wc appears in each list and places these in a dictionary

    total_occurrences = dict(Counter(all_actual_classes)) #Counter sorts them so {1(nouns):300 times, 2(verb):150 times..etc)
    correct_counts = Counter(corr_actual_class)
    wrong_counts = Counter(wrong_predicted_class)

    #Sorts them
    total_occurrences = {k: total_occurrences[k] for k in sorted(total_occurrences)}
    correct_counts = {k: correct_counts[k] for k in sorted(correct_counts)}
    wrong_counts = {k: wrong_counts[k] for k in sorted(wrong_counts)}

    print(total_occurrences)
    print(correct_counts)
    print(wrong_counts)


    plot_missed(correct_counts, wrong_counts)
    plot_found(correct_counts, total_occurrences)

def plot_missed(correct, incorrect):
    x_right = []
    x_left = []
    x = []
    for x_values in correct.keys():
        x_right.append((x_values+0.15))
    for x_values in incorrect.keys():
        x_left.append(x_values-0.15)
        x.append(x_values)
    plt.bar(x_left, incorrect.values(), 0.3, label='Incorrect')
    plt.bar(x_right, correct.values(), 0.3, label='Correct')
    plt.title('Correct vs incorrect predictions for a certain class')

    plt.xticks(x)
    plt.legend()
    plt.show()

def plot_found(correct, total_occurrences):
    x_right = []
    x_left = []
    x = []
    for x_values in correct.keys():
        x_right.append((x_values+0.15))
    for x_values in total_occurrences.keys():
        x_left.append(x_values-0.15)
        x.append(x_values)
    plt.bar(x_left, total_occurrences.values(), 0.3, label='Total amount in test data')
    plt.bar(x_right, correct.values(), 0.3, label='Correctly identified')
    plt.title('Correct identification of each word class out of total')
    plt.xticks(x)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass


