import json

import numpy as np
from dataProcessing import open_dict

class_to_index_wait1 = {
    """ Probably all word classes, but if there are others they will show up when we get errors"""
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    'AB': 5,
    'PM': 6,
    '.': 7,
    'DT': 8,
    'KN': 9,
    'PC': 10,
    'SN': 11,
    'PS': 12,
    'PN': 13,
    'HP': 14,
    'IE': 15,
    'RO': 16,
    'HA': 17,
    'PAD': 18,
    'PL': 19,
    'MAD': 20,
    'UO' : 21,
    'HD' : 22,
    'RG' : 23,
    'MID' : 24,
    'IN' : 25,
    'HS' : 26
    }
class_to_index_wait2 = {
    #Lite motivationer, kommer raffineras
    #PN = PM = PS = IN <=> pronomen = egennamn dvs Han = Stockholm = hans = AJ!
    #HA = HP <=> frågande pronomen = frågande adverb dvs vem = när
    #NA = UO (utländskt ord)
    # MAD = . = PAD = MID? dvs . = . = .,; (just nu iaf då vi filtrerar bort nästan allt)
    #  HS= något HS =vars, vems osv
    # PL = nåt finns bara en i datan, kollade i classdict och hittade exemplet "tillinitiativ" som en enskild sträng??
    #  RG= RO två = andra
    # HD (relativt besätmning) exemplet från classdict är "hurdana"??
    # SN subjunktion exemplet från classdict är "50som"??
    # IE  verkar vara tom
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    'AB': 5,
    'PN': 6,
    '.': 7,
    'DT': 8,
    'KN': 9,
    'PC': 10,
    'SN': 11,
    'HP': 12,
    'RO': 13,
    'PS': 6,
    'PM': 1,
    'HA': 12,
    'PAD': 7,
    'PL': 0,
    'MAD': 7,
    'UO' : 0,
    'HD' : 12,
    'RG' : 13,
    'MID' : 7,
    'IN' : 6,
    'HS' : 12
}

class_to_index = {
    # Pronomen = substantiv Han är där = Grejen är där
    # Adjektiv = Adverb Han är glad, Han springer fort
    # konjunktion = preposition ( gissar lite )
    # particip = adjektiv "Particip är ordformer som utgår från verb, men fungerar som adjektiv. "(källaisof)
    # Subjunktion = konjunktion (båda binder ihop satsdelar)
    # HP = pronomen (vem är där? = han är där)
    # RO = adjektiv (Han är först, han är på andra plats = Han är bäst, han är på sämsta platsen) lite oklart men kanske
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    '.': 5,
    'DT': 6,
    'AB': 4,
    'PM': 1,
    'KN': 3,
    'PC': 4,
    'SN': 3,
    'HP': 1,
    'RO': 4,
    'PS': 1,
    'PN': 1,
    'HA': 1,
    'PAD': 5,# ev Problem
    'PL': 0,
    'MAD': 5, # ev Problem
    'UO' : 0,
    'HD' : 1,
    'RG' : 4,
    'MID' : 5, # ev problem
    'IN' : 1,
    'HS' : 1
}

def iterate_transition_matrix(word_classes):
    # Creating an empty matrix
    transition_matrix = []
    mat_size = max(class_to_index.values()) + 1

    for _ in range(mat_size):
        transition_matrix.append([0] * mat_size)

    for i in range(len(word_classes) - 1):
        # indexes written out a bit
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[current_index][next_index] += 1

    # Converting to a probabilty matrix (all rows sum to 1)
    for i in range(mat_size): # for some row
        n = sum(transition_matrix[i]) # summing the row
        if n>0:
            for j in range(mat_size): # for element in the row
                transition_matrix[i][j] = transition_matrix[i][j]/n # normalizing
    return transition_matrix

def run_1_order(file, t_matrix_name):
    word_classes = open_dict(file)
    transition_matrix = iterate_transition_matrix(word_classes)
    with open(t_matrix_name, "w") as outfile:
        json.dump(transition_matrix, outfile)
    return transition_matrix


def iterate_tm_2_order(word_classes):
    mat_size = max(class_to_index.values()) + 1
    # Creating an empty matrix
    transition_matrix = np.zeros((mat_size, mat_size, mat_size))
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
    print(transition_matrix)
    for i in range(mat_size):  # for some row
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            n = sum(transition_matrix[i][k])  # summing this row
            if n > 0:
                for j in range(mat_size):  # for element in the row
                    transition_matrix[k][i][j] = transition_matrix[k][i][j] / n  # normalizing
    return transition_matrix


def run_2_order(file, t_matrix_name):
    word_classes = open_dict(file)
    tm_2nd_order = iterate_tm_2_order(word_classes)
    tm_2nd_order = tm_2nd_order.tolist()
    with open(t_matrix_name, "w") as outfile:
        json.dump(tm_2nd_order, outfile)
    return tm_2nd_order


def iterate_tm_3_order(word_classes):
    mat_size = max(class_to_index.values()) + 1
    # Creating an empty matrix
    transition_matrix = np.zeros((mat_size, mat_size, mat_size, mat_size))
    for i in range(2, len(word_classes) - 1):
        # indexes written out a bit
        old_2_class = word_classes[i-2]
        old_class = word_classes[i-1]
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        old_2_index = class_to_index[old_2_class]
        old_index = class_to_index[old_class]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[old_2_index][old_index][current_index][next_index] += 1

    for i in range(mat_size):  # for some row
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            for p in range(mat_size):
                n = sum(transition_matrix[p][k][i])  # summing this row
                if n > 0:
                    for j in range(mat_size):  # for element in the row
                        transition_matrix[p][k][i][j] = transition_matrix[p][k][i][j] / n  # normalizing
    return transition_matrix

def run_3_order(file, t_matrix_name):
    word_classes = open_dict(file)
    tm_3rd_order = iterate_tm_3_order(word_classes)
    tm_3rd_order = tm_3rd_order.tolist()
    with open(t_matrix_name, "w") as outfile:
        json.dump(tm_3rd_order, outfile)
    return tm_3rd_order


def iterate_transition_matrix_future(word_classes):
    # Creating an empty matrix
    transition_matrix = []
    mat_size = max(class_to_index.values()) + 1

    for _ in range(mat_size):
        transition_matrix.append([0] * mat_size)

    for i in range(len(word_classes) - 1):
        # indexes written out a bit
        current_class = word_classes[i+1]
        next_class = word_classes[i]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation
        transition_matrix[current_index][next_index] += 1

    # Converting to a probabilty matrix (all rows sum to 1)
    for i in range(mat_size): # for some row
        n = sum(transition_matrix[i]) # summing the row
        if n>0:
            for j in range(mat_size): # for element in the row
                transition_matrix[i][j] = transition_matrix[i][j]/n # normalizing
    return transition_matrix

def run_1_order_future(file, t_matrix_name):
    word_classes = open_dict(file)
    transition_matrix = iterate_transition_matrix_future(word_classes)
    with open(t_matrix_name, "w") as outfile:
        json.dump(transition_matrix, outfile)
    return transition_matrix


if __name__ == "__main__":
    #run_1_order('wordclasslists/WC_all.json', "transition_matrices/TM_all")
    """run_1_order('wordclasslists/WC_transl.json', "transition_matrices/TM_transl.json")
    run_1_order('wordclasslists/WC_non_transl.json', 'transition_matrices/TM_non_transl.json')
    run_2_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_2nd')
    run_3_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_3rd')"""
    run_1_order_future("wordclasslists/WC_all.json", "transition_matrices/TM_all_future")