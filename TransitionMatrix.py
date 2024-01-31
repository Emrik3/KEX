#To do: 1.Wait for better word_classes list to be better
#2. implement iteration over many lists/ doing this but with a long list

import json

# Ordered word classes from 1 abstract (placeholder)
word_classes = ['NA', 'NA', 'NA', 'VB', 'PN', 'JJ', 'NN', 'NA', 'KN', 'AB', 'VB', 'RG', 'JJ', 'NA', 'PP', 'JJ', 'NN',
                'PP', 'JJ', 'NA', '.', 'PN', 'NA', 'NA', 'VB', 'NA', 'NA', 'KN', 'AB', 'NA', 'PP', 'JJ', 'NA', '.',
                'NN', 'PP', 'NA', 'VB', 'IE', 'AB', 'NA', 'KN', 'JJ', 'NN', 'VB', 'PN', 'PP', 'IE', 'VB', 'JJ', 'NA',
                '.', 'PN', 'VB', 'AB', 'IE', 'AB', 'NA', 'NN', 'PP', 'NA', 'NA', 'PP', 'NA', 'KN', 'NA', 'PP', 'IE',
                'NA', 'NA', 'AB', 'VB', 'JJ', 'NA', '.', 'NN', 'NA', 'NN', 'KN', 'VB', 'VB', 'NN', 'PP', 'NA', 'KN',
                'NA', 'NN', 'KN', 'NA', 'NN', 'PP', 'NA', '.', 'NA', 'VB', 'VB', 'IE', 'VB', 'JJ', 'KN', 'AB', 'IE',
                'NA', '.', 'NN', 'PP', 'NN', 'KN', 'NN', 'PP', 'NA', 'VB', 'AB', 'AB', 'PL', 'PP', 'NN', '.', 'DT',
                'NA', 'AB', 'PP', 'RO', 'NN', 'NA', 'KN', 'NA', 'NA', '.']

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
    'IN' : 25
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

def create_and_calculate():
    word_classes = open_dict('word_classes.json')
    dimensioned_matrix = create_transition_matrix(class_to_index)
    transition_matrix = iterate_transition_matrix(word_classes, dimensioned_matrix)
    print(transition_matrix)
    with open("transition_matrix.json", "w") as outfile:
        json.dump(transition_matrix, outfile)

create_and_calculate()