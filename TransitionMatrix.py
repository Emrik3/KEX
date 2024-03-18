import json

import numpy as np
from dataProcessing import open_dict, read_traning_csv, read_translation_txt, text_cleaner, save_dict, check_english
from choose_word_classes import class_to_index, ending_list, ending_to_num
import random



def iterate_transition_matrix(word_classes, setup, mixed):
    if mixed:
        random.shuffle(word_classes)
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
        if setup == [0, 1]:
            transition_matrix[current_index][next_index] += 1
        if setup == [1, 0]:
        # backward
            transition_matrix[next_index][current_index] += 1

    # Converting to a probabilty matrix (all rows sum to 1)
    for i in range(mat_size): # for some row
        n = sum(transition_matrix[i]) # summing the row
        if n>0:
            for j in range(mat_size):  # for element in the row
                transition_matrix[i][j] = transition_matrix[i][j]/n # normalizing
    return transition_matrix

def run_1_order(word_classes, t_matrix_name,setup, mixed):
    if type(word_classes) != list:
        word_classes = open_dict(word_classes)
    transition_matrix = iterate_transition_matrix(word_classes, setup, mixed)
    np.save(t_matrix_name, transition_matrix)
    return transition_matrix


def iterate_tm_2_order(word_classes, setup, mixed):
    if mixed:
        random.shuffle(word_classes)
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
        if setup == [0, 0, 1]:
            transition_matrix[old_index][current_index][next_index] += 1
        if setup == [0, 1, 0]:
        # looks at the probability to get current wc given the old wc and next wc
            transition_matrix[old_index][next_index][current_index] += 1
        if setup == [1, 0, 0]:
            transition_matrix[current_index][next_index][old_index] += 1
    for i in range(mat_size):  # for some row
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            n = sum(transition_matrix[i][k])  # summing this row
            if n > 0:
                for j in range(mat_size):  # for element in the row
                    transition_matrix[i][k][j] = transition_matrix[i][k][j] / n  # normalizing
    return transition_matrix


def run_2_order(word_classes, t_matrix_name, setup, mixed):
    if type(word_classes) != list:
        word_classes = open_dict(word_classes)
    tm_2nd_order = iterate_tm_2_order(word_classes, setup, mixed)
    np.save(t_matrix_name, tm_2nd_order)  # Avoid json and .tolist() since they are slow
    return tm_2nd_order


def iterate_tm_3_order(word_classes,setup, mixed):
    if mixed:
        random.shuffle(word_classes)
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
        if setup == [0, 0, 0, 1]:
            transition_matrix[old_2_index][old_index][current_index][next_index] += 1
        if setup == [0, 0, 1, 0]:
            transition_matrix[old_2_index][old_index][next_index][current_index] += 1
        if setup == [0, 1, 0, 0]:
            transition_matrix[old_2_index][current_index][next_index][old_index] += 1
        if setup == [1, 0, 0, 0]:
            transition_matrix[old_index][current_index][next_index][old_2_index] += 1


    for i in range(mat_size):  # for some row
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            for p in range(mat_size):
                n = sum(transition_matrix[p][k][i])  # summing this row
                if n > 0:
                    for j in range(mat_size):  # for element in the row
                        transition_matrix[i][k][p][j] = transition_matrix[i][k][p][j] / n  # normalizing
    return transition_matrix

def run_3_order(word_classes, t_matrix_name,setup, mixed):
    if type(word_classes) != list:
        word_classes = open_dict(word_classes)
    tm_3rd_order = iterate_tm_3_order(word_classes,setup, mixed)
    np.save(t_matrix_name, tm_3rd_order)
    return tm_3rd_order

def iterate_tm_4_order(word_classes, setup, mixed):
    if mixed:
        random.shuffle(word_classes)
    mat_size = max(class_to_index.values()) + 1
    # Creating an empty matrix
    transition_matrix = np.zeros((mat_size, mat_size, mat_size, mat_size, mat_size))
    for i in range(3, len(word_classes) - 1):
        # indexes written out a bit
        old_3_class = word_classes[i-3]
        old_2_class = word_classes[i-2]
        old_class = word_classes[i-1]
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        old_3_index = class_to_index[old_3_class]
        old_2_index = class_to_index[old_2_class]
        old_index = class_to_index[old_class]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation; Need this explained not sure if it is correct
        if setup == [0, 0, 0, 0, 1]:
            transition_matrix[old_3_index][old_2_index][old_index][current_index][next_index] += 1
        if setup == [0, 0, 0, 1, 0]:
            transition_matrix[old_3_index][old_2_index][old_index][next_index][current_index] += 1
        if setup == [0, 0, 1, 0, 0]:
            transition_matrix[old_3_index][old_2_index][current_index][next_index][old_index] += 1
        if setup == [0, 1, 0, 0, 0]:
            transition_matrix[old_3_index][old_index][current_index][next_index][old_2_index] += 1
        if setup == [1, 0, 0, 0, 0]:
            transition_matrix[old_2_index][old_index][current_index][next_index][old_3_index] += 1

    for i in range(mat_size):  # for some row
        print(i)
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            for p in range(mat_size):
                for q in range(mat_size):
                    n = sum(transition_matrix[q][p][k][i])  # summing this row
                    if n > 0:
                        for j in range(mat_size):  # for element in the row
                            transition_matrix[i][k][p][q][j] = transition_matrix[i][k][p][q][j] / n  # normalizing
    return transition_matrix

def run_4_order(word_classes, t_matrix_name, setup, mixed):
    if type(word_classes) != list:
        word_classes = open_dict(word_classes)
    tm_4th_order = iterate_tm_4_order(word_classes, setup, mixed)
    np.save(t_matrix_name, tm_4th_order)
    return tm_4th_order

def iterate_tm_5_order(word_classes, setup):
    mat_size = max(class_to_index.values()) + 1
    # Creating an empty matrix
    transition_matrix = np.zeros((mat_size, mat_size, mat_size, mat_size, mat_size, mat_size))
    for i in range(4, len(word_classes) - 1):
        # indexes written out a bit
        old_4_class = word_classes[i-4]
        old_3_class = word_classes[i-3]
        old_2_class = word_classes[i-2]
        old_class = word_classes[i-1]
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        old_4_index = class_to_index[old_4_class]
        old_3_index = class_to_index[old_3_class]
        old_2_index = class_to_index[old_2_class]
        old_index = class_to_index[old_class]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation; Need this explained not sure if it is correct
        if setup == [0, 0, 0, 0, 0, 1]:
            transition_matrix[old_4_index][old_3_index][old_2_index][old_index][current_index][next_index] += 1
        if setup == [0, 0, 0, 0, 1, 0]:
            transition_matrix[old_4_index][old_3_index][old_2_index][old_index][next_index][current_index] += 1
        if setup == [0, 0, 0, 1, 0, 0]:
            transition_matrix[old_4_index][old_3_index][old_2_index][current_index][next_index][old_index] += 1
        if setup == [0, 0, 1, 0, 0, 0]:
            transition_matrix[old_4_index][old_3_index][old_index][current_index][next_index][old_2_index] += 1
        if setup == [0, 1, 0, 0, 0, 0]:
            transition_matrix[old_4_index][old_2_index][old_index][current_index][next_index][old_3_index] += 1
        if setup == [1, 0, 0, 0, 0, 0]:
            transition_matrix[old_3_index][old_2_index][old_index][current_index][next_index][old_4_index] += 1

    for i in range(mat_size):  # for some row
        print(i)
        for k in range(mat_size): # every row has another row in the "new" direction because 3d
            for p in range(mat_size):
                for q in range(mat_size):
                    for z in range(mat_size):
                        n = sum(transition_matrix[z][q][p][k][i])  # summing this row
                        if n > 0:
                            for j in range(mat_size):  # for element in the row
                                transition_matrix[i][k][p][q][z][j] = transition_matrix[i][k][p][q][z][j] / n  # normalizing
    return transition_matrix

def run_5_order(file, t_matrix_name, setup):
    word_classes = open_dict(file)
    tm_5th_order = iterate_tm_5_order(word_classes, setup)
    print("here")
    np.save(t_matrix_name, tm_5th_order) # Avoid json and .tolist() since they are slow

    print("exit")
    return tm_5th_order

def iterate_tm_6_order(word_classes, setup):
    mat_size = max(class_to_index.values()) + 1
    # Creating an empty matrix
    transition_matrix = np.zeros((mat_size, mat_size, mat_size, mat_size, mat_size, mat_size, mat_size))
    for i in range(5, len(word_classes) - 1):
        # indexes written out a bit
        old_5_class = word_classes[i-5]
        old_4_class = word_classes[i - 4]
        old_3_class = word_classes[i - 3]
        old_2_class = word_classes[i - 2]
        old_class = word_classes[i - 1]
        current_class = word_classes[i]
        next_class = word_classes[i + 1]
        old_5_index = class_to_index[old_5_class]
        old_4_index = class_to_index[old_4_class]
        old_3_index = class_to_index[old_3_class]
        old_2_index = class_to_index[old_2_class]
        old_index = class_to_index[old_class]
        current_index = class_to_index[current_class]
        next_index = class_to_index[next_class]

        # The calculation; Need this explained not sure if it is correct
        if setup == [0, 0, 0, 0, 0, 0, 1]:
            transition_matrix[old_4_index][old_3_index][old_2_index][old_index][current_index][next_index] += 1
        if setup == [0, 0, 0, 0, 0, 1, 0]:
            transition_matrix[old_4_index][old_3_index][old_2_index][old_index][next_index][current_index] += 1
        if setup == [0, 0, 0, 0, 1, 0, 0]:
            transition_matrix[old_4_index][old_3_index][old_2_index][current_index][next_index][old_index] += 1
        if setup == [0, 0, 0, 1, 0, 0, 0]:
            transition_matrix[old_4_index][old_3_index][old_index][current_index][next_index][old_2_index] += 1
        if setup == [0, 0, 1, 0, 0, 0, 0]:
            transition_matrix[old_4_index][old_2_index][old_index][current_index][next_index][old_3_index] += 1
        if setup == [0, 1, 0, 0, 0, 0, 0]:
            transition_matrix[old_3_index][old_2_index][old_index][current_index][next_index][old_4_index] += 1
        if setup == [1, 0, 0, 0, 0, 0, 0]:
            transition_matrix[old_4_index][old_3_index][old_2_index][old_index][current_index][next_index][old_5_index] += 1

    for i in range(mat_size):  # for some row
        #To slow at the moment, needs new approach
        print(i)
        for k in range(mat_size):  # every row has another row in the "new" direction because 3d
            for p in range(mat_size):
                for q in range(mat_size):
                    for z in range(mat_size):
                        for v in range(mat_size):
                            n = sum(transition_matrix[z][q][p][k][i][v])  # summing this row
                            if n > 0:
                                for j in range(mat_size):  # for element in the row
                                    transition_matrix[i][k][p][q][z][v][j] = transition_matrix[i][k][p][q][z][v][
                                                                          j] / n  # normalizing
    return transition_matrix

def run_6_order(file, t_matrix_name, setup):
    word_classes = open_dict(file)
    tm_6th_order = iterate_tm_6_order(word_classes, setup)
    print("here")
    np.save(t_matrix_name, tm_6th_order) # Avoid json and .tolist() since they are slow
    return tm_6rd_order

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


def ending_freq(text, ending_list):
    # Saves dictionary of probability of ending, does not look at one letter words
    ending_freq = {}
    #text = text_cleaner(text)
    for ending in ending_list:
        ending_freq[ending] = 0
    ending_prob = ending_freq
    words = text.split(' ')
    for word in words:
        if len(word) > 1:
            ending_freq[word[-2:]] += 1
    total_words = sum(ending_freq.values())
    for ending in ending_list:
        ending_prob[ending] = ending_freq[ending] / total_words

    ending_prob = dict(sorted(ending_prob.items(), key=lambda item: item[1], reverse=True))
    json_object = json.dumps(ending_prob, indent=4)
    with open("dictionaries/ending_prob.json", "w") as outfile:
        outfile.write(json_object)

def prob_ending_class(giventext, word_class_dict):
    # Calculates the probability of a certian wordending to be a certian wordclass
    wcdict = open_dict(word_class_dict)
    text = read_traning_csv(giventext)
    res = []
    i=0
    textlist = []
    for textt in text:
        textt = text_cleaner(textt, no_dot=False)
        sentences = textt.lower().split('. ')
        for sentence in sentences:
            words = sentence.split(' ')
            textlist.append(words)


    ending_matrix = np.zeros((len(ending_list), len(class_to_index.keys())))
    for i in range(len(textlist)):
        for j in range(len(textlist[i])):
            if len(textlist[i][j]) >= 2 and textlist[i][j] in list(wcdict.keys()) and textlist[i][j][-2:] in ending_list:
                ending_matrix[ending_to_num[textlist[i][j][-2:]]][class_to_index[wcdict[textlist[i][j]]]] += 1

    for i in range(len(ending_list)): # for some row
        n = sum(ending_matrix[i]) # summing the row
        if n > 0:
            for j in range(len(class_to_index.keys())):  # for element in the row
                ending_matrix[i][j] = ending_matrix[i][j]/n # normalizing

    np.save("wordclasslists/WCending", ending_matrix)
        
    return ending_matrix


def emmision_matrix(file, t_matrix_name):
    """possibly to implement in the future"""


if __name__ == "__main__":
    print(prob_ending_class())
    #run_1_order('wordclasslists/WC_all.json', "transition_matrices/TM_all")
    #run_1_order('wordclasslists/WC_transl.json', "transition_matrices/TM_transl.json")
    #run_1_order('wordclasslists/WC_non_transl.json', 'transition_matrices/TM_non_transl.json')
    #run_2_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_2nd')
    #run_3_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_3rd')
    #run_1_order_future("wordclasslists/WC_all.json", "transition_matrices/TM_all_future")
    #text = read_translation_txt('Trainingdata/translated_sample.txt')
    #ending_freq(text, ending_list)
