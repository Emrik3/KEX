import json

import dataProcessing
from dataProcessing import *

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

def create_transition_matrix(class_to_index):
    # Creating an empty matrix
    transition_matrix = []
    for _ in range(len(class_to_index)):
        transition_matrix.append([0] * len(class_to_index))
    return transition_matrix

def iterate_transition_matrix(word_classes, transition_matrix):
    for i in range(len(word_classes) - 1):
        # indexes written out a bit
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[current_index][next_index] += 1

    # Converting to a probabilty matrix (all rows sum to 1)
    for row in transition_matrix:
        n = sum(row)
        if n > 0:
            row[:] = [f / sum(row) for f in row]
    return transition_matrix

def create_and_calculate(file, t_matrix_name):
    word_classes = open_dict(file)
    dimensioned_matrix = create_transition_matrix(class_to_index)
    transition_matrix = iterate_transition_matrix(word_classes, dimensioned_matrix)
    with open(t_matrix_name, "w") as outfile:
        json.dump(transition_matrix, outfile)
    return transition_matrix

if __name__ == "__main__":
    create_and_calculate('WC_all.json', "TM_all.json")
    create_and_calculate('WC_transl.json', "TM_transl.json")
    create_and_calculate('WC_non_transl.json', 'TM_non_transl.json')