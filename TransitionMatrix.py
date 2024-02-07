import json

import dataProcessing
from dataProcessing import *
import numpy as np
# Probably all word classes, but if there are others they will show up when we get errors
class_to_index = {
    'NA': 0,
    'NN': 1,
    'HP': 2,
    'VB': 3,
    'MAD': 4,
    'PP': 5,
    'AB': 6,
    'MID': 7,
    'DT': 8,
    'RG': 9,
    'JJ': 10,
    'KN': 11,
    'PM': 12,
    'PC': 13,
    'IE': 14,
    'PN': 15,
    'RO': 16,
    'HA': 17,
    'PAD': 18,
    'PL': 19,
    '.': 20,
    'UO' : 21,
    'HD' : 22,
    'SN' : 23,
    'PS' : 24,
    'IN' : 25,
    'HS' : 26
    }


def open_dict(file):
    with open(file, 'r') as openfile:
        # Reading from json file
        return json.load(openfile)

def iterate_transition_matrix(word_classes):
    # Creating an empty matrix
    transition_matrix = []
    for _ in range(len(class_to_index)):
        transition_matrix.append([0] * len(class_to_index))

    for i in range(len(word_classes) - 1):
        # indexes written out a bit
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[current_index][next_index] += 1

    # Converting to a probabilty matrix (all rows sum to 1)
    #made this more clear now
    for i in range(len(class_to_index)): # for some row
        n = sum(transition_matrix[i]) # summing the row
        if n>0:
            for j in range(len(class_to_index)): # for element in the row
                transition_matrix[i][j] = transition_matrix[i][j]/n # normalizing
    return transition_matrix

def create_and_calculate(file, t_matrix_name):
    word_classes = open_dict(file)
    transition_matrix = iterate_transition_matrix(word_classes)
    with open(t_matrix_name, "w") as outfile:
        json.dump(transition_matrix, outfile)
    return transition_matrix

#For a 2d markov chain
def iterate_tm_2_order(word_classes):
    # Creating an empty matrix
    transition_matrix = np.zeros((len(class_to_index), len(class_to_index), len(class_to_index)))
    for i in range(1, len(word_classes) - 1):
        # indexes written out a bit
        old_class = word_classes[i-1]
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        old_index = class_to_index[old_class]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[old_index][current_index][next_index] += 1

    for i in range(len(class_to_index)):  # for some row
        for k in range(len(class_to_index)): # every row has another row in the "new" direction because 3d
            n = sum(transition_matrix[i][k])  # summing this row
            if n > 0:
                for j in range(len(class_to_index)):  # for element in the row
                    transition_matrix[k][i][j] = transition_matrix[k][i][j] / n  # normalizing
    return transition_matrix

def run_2_order(file, t_matrix_name):
    word_classes = open_dict(file)
    tm_2nd_order = iterate_tm_2_order(word_classes)
    tm_2nd_order = tm_2nd_order.tolist()
    with open(t_matrix_name, "w") as outfile:
        json.dump(tm_2nd_order, outfile)
    return tm_2nd_order

if __name__ == "__main__":
    #create_and_calculate('WC_all.json', "TM_all.json")
    #create_and_calculate('WC_transl.json', "TM_transl.json")
    #create_and_calculate('WC_non_transl.json', 'TM_non_transl.json')
    run_2_order('WC_all.json', 'TM_all_2nd')