# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index
from collections import Counter
import numpy as np
from choose_word_classes import number_to_class
import pandas as pd



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
def plot_table(data, columns, rows):
    n_rows = len(data) # antal ordklasser
    y_offset = np.zeros(len(columns))
    cell_text = []
    for row in range(n_rows):
        y_offset = data[row]
        cell_text.append([x for x in y_offset])
    # Reverse text labels to display
    cell_text.reverse()

    # Add a table at the bottom of the axes
    plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()
def plot_line_graph(data, columns, rows, offset):
    # Get some pastel shades for the colors
    n_rows = len(data)
    index = np.arange(len(columns)) + 0.3
    for row in range(n_rows):
        plt.plot(index, data[row], label=str(number_to_class[row+offset]))
    plt.xticks(np.arange(len(columns)),columns)
    plt.legend()
    plt.title('Metrics for each word class')

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
    confusion_metrics(confusionmatrix)

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


    
def confusion_metrics(matrix):
    """Calculates 5 metrics using the confusion matrix"""
    data = []
    for i in range(len(matrix)): # for some row
        FN = [0] * len(class_to_index)
        TP = [0] * len(class_to_index)
        FP = [0] * len(class_to_index)
        TN = [0] * len(class_to_index)
        counter = 0
        for j in range(len(matrix)): # for element in row
            if i==j:
                TP[i] += matrix[i][j] # diagonal elements give TP
            else:
                FN[i] += matrix[i][j] # sums every other element in the row
                FP[i] += matrix[j][i] # sums every other element in the column
            TN[i] = np.sum(matrix)-(TP[i] + FN[i] + FP[i]) # sum of all other values are true negatives
        if (TP[i] + TN[i] + FP[i] + FN[i]) >0: #divide by zero check
            Accuracy = round((TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]),3)
            #print("Accuracy for word class " + str((number_to_class[i])) + "  : " + str(Accuracy*100) + "%")
            counter+=1
        else:
            Accuracy = 0.0
        if (TP[i] + FP[i])>0: #divide by zero check
            Precision = round((TP[i]) / (TP[i] + FP[i]),3)
            #print("Precision for word class " + str((number_to_class[i])) + "  : " + str(Precision*100) + "%")
            counter +=1
        else:
            Precision = 0.0
        if (TN[i] + FP[i])>0: #divide by zero check
            Recall = round((TP[i]) / (TP[i] + FN[i]),3)
            #print("Recall for word class " + str((number_to_class[i])) + "  : " + str(Recall*100))
        else:
            Recall = 0.0
        if counter==2: #divide by zero check
            F1_score = round((2*Precision*Recall)/(Precision+Recall),3)
            #print("F1 score for word class " + str((number_to_class[i])) + "  : " + str(100 * F1_score) + "%")
        else:
            F1_score = 0.0
        if (TN[i] + FP[i])>0 and TN[i]>0: #divide by zero check
            Specificity = round((TN[i]) / (TN[i] + FP[i]), 3)
            #print("Specificity for word class " + str((number_to_class[i])) + "  : " + str(100*Specificity) + "%")
        else:
            Specificity = 0.0
        data.append([Accuracy, Precision, Recall, F1_score, Specificity])

    columns = ('Accuracy', 'Precision', 'Recall', 'F1 score', 'Specificity')
    wc_list = []
    for wc in class_to_index.keys():
        wc_list.append(wc)
    rows = wc_list # y ticks for table
    data.reverse()
    print(data)
    print(columns)
    print(rows)



    plot_table(data, columns, rows)
    plot_line_graph(data[0:8], columns, rows[0:8], offset=0)
    plot_line_graph(data[9:16], columns, rows[9:16], offset=9)
    plot_line_graph(data[17:], columns, rows[17:], offset=17)




    
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


