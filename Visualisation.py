# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index
from collections import Counter
import numpy as np


def transition_matrix_vis(matrix):
    """Generates heat map of transition matrix """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.flipud(matrix), cmap='viridis', interpolation='nearest')
    fig.colorbar(im)
    mat_size = max(class_to_index.values()) + 1
    ax.set_xticks(range(mat_size))
    ax.set_yticks(range(mat_size, 0, -1))

    keys_to_include = list(class_to_index.keys())[0:mat_size]
    ax.set_xticklabels(keys_to_include)
    ax.set_yticklabels(keys_to_include)
    ax.tick_params(axis='both', which='major', labelsize=10) # Set fontsize on labels, this is hard to get right cause dont fit.
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.show()

def organize_and_plot(res, order):
    wrong_predicted_class = [] # When the model predicted wrong it predicted these wc
    wrong_actual_class = [] # When the model predicted wrong it should have predicted these wc
    corr_actual_class = [] # When the model predicted right it predicted these wc
    confusionmatrix = np.zeros((len(class_to_index), len(class_to_index)))
    print("res: " + str(len(res)))
    for elem in res:
        wrong_predicted_class.append(elem[0])
        wrong_actual_class.append(elem[1])
        corr_actual_class.append(elem[2])
        confusionmatrix += elem[3]
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

    print("total: " + str(total_occurrences))
    print("correct: " + str(correct_counts))
    print("wrong: " + str(wrong_counts))
    try:
        non_guess = wrong_counts[0]
    except:
        non_guess = 0
    tot_correct = sum(list(correct_counts.values()))
    tot_tot = sum(list(total_occurrences.values()))
    tot_incorrect = sum(list(total_occurrences.values()))

    print("non_guess: "+ str(non_guess) + ", tot_correct: " + str(tot_correct) + ", total_total: " + str(tot_tot) + ", incorrect: " + str(tot_incorrect))
    print("Right predictions out of total: " + (str(tot_correct/tot_tot)))
    print("Right predictions out of all made predictions: " + str((tot_correct)/(tot_tot-non_guess)))
    transition_matrix_vis(confusionmatrix)
    plot_missed(correct_counts, wrong_counts, order)
    plot_found(correct_counts, total_occurrences, order)

    
def plot_missed(correct, incorrect, order):
    x_right = []
    x_left = []
    x = []
    for x_values in correct.keys():
        x_right.append((x_values+0.15))
    for x_values in incorrect.keys():
        x_left.append(x_values-0.15)
        x.append(x_values)
    print(x_left)
    print(x_right)
    print(incorrect.values())
    print(correct.values())
    plt.bar(x_left, list(incorrect.values()), 0.3, label='Incorrect')
    plt.bar(x_right, list(correct.values()), 0.3, label='Correct')
    plt.title('Correct vs Incorrect Predictions by Markov Chain Order: ' + str(order)) 
    plt.xlabel("Word class")
    plt.ylabel("Predicted word classes")
    plt.xticks(x)
    plt.legend()
    print("hej")
    plt.show()

def plot_found(correct, total_occurrences, order):
    x_right = []
    x_left = []
    x = []
    for x_values in correct.keys():
        x_right.append((x_values+0.15))
    for x_values in total_occurrences.keys():
        x_left.append(x_values-0.15)
        x.append(x_values)
    plt.bar(x_left, total_occurrences.values(), 0.3, label='Total amount')
    plt.bar(x_right, correct.values(), 0.3, label='Correctly identified by model')
    plt.title('Correct Identification vs Total by Markov Chain Order: ' + str(order))
    plt.xlabel("Word class")
    plt.ylabel("Word classes in data")
    plt.xticks(x)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    pass


