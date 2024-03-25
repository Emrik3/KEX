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
    index = np.arange(len(columns))
    for row in range(len(rows)):
        plt.plot(index, data[row+offset], label=str(number_to_class[row+offset]))
    plt.xticks(np.arange(len(columns)),columns)
    plt.legend()
    plt.title('Metrics for each word class')
    plt.show()
def organize_and_plot(res, order, setup, plot):
    wrong_predicted_class = [] # When the model predicted wrong it predicted these wc
    wrong_actual_class = [] # When the model predicted wrong it should have predicted these wc
    corr_actual_class = [] # When the model predicted right it predicted these wc
    confusionmatrix = np.zeros((len(class_to_index), len(class_to_index)))
    #print("res: " + str(len(res)))
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

    #print("total: " + str(total_occurrences))
    #print("correct: " + str(correct_counts))
    #print("wrong: " + str(wrong_counts))
    try:
        non_guess = wrong_counts[0]
    except:
        non_guess = 0
    tot_correct = sum(list(correct_counts.values()))
    tot_tot = sum(list(total_occurrences.values()))
    tot_incorrect = sum(list(total_occurrences.values()))

    #print("non_guess: "+ str(non_guess) + ", tot_correct: " + str(tot_correct) + ", total_total: " + str(tot_tot) + ", incorrect: " + str(tot_incorrect))
    if plot:
        print("Right predictions to incorrect ratio, total: " + (str(tot_correct/tot_incorrect)))
        print("Right predictions out of all made predictions: " + str((tot_correct)/(tot_tot-non_guess)))
        confusion_metrics(confusionmatrix,setup)
        transition_matrix_vis(confusionmatrix)
    #plot_missed(correct_counts, wrong_counts, total_occurrences, order, setup, 100*tot_correct/tot_tot)
    return tot_correct/tot_tot

    
def confusion_metrics(matrix,setup):
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
        if (TP[i] + TN[i] + FP[i] + FN[i]) >0 and ((TP[i] + TN[i])) >0: #divide by zero check
            Accuracy = round((TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]),3)
            #print("Accuracy for word class " + str((number_to_class[i])) + "  : " + str(Accuracy*100) + "%")
            counter+=1
        else:
            Accuracy = 0.0
        if (TP[i] + FP[i])>0 and TP[i]>0: #divide by zero check
            Precision = round((TP[i]) / (TP[i] + FP[i]),3)
            #print("Precision for word class " + str((number_to_class[i])) + "  : " + str(Precision*100) + "%")
            counter +=1
        else:
            Precision = 0.0
        if (TN[i] + FP[i])>0 and TP[i]>0: #divide by zero check
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
    data.reverse()
    plot_line_graph(data, columns, rows[0:5], offset=1)
    plot_confusion_metric_bar([item[3] for item in data],[item[1] for item in data],[item[2] for item in data],rows, setup) #Accuracy, Precision Recall
   # plot_confusion_metric_bar([item[3] for item in data],[item[4] for item in data],None,rows, setup) #F1, specificity

    #plot_confusion_metric_bar([item[3] for item in data], rows, "F1 score") #F1 score
    #plot_confusion_metric_bar([item[4] for item in data], rows, "Specificity") # Specificity

def plot_missed(correct, incorrect, total, order, setup, perc):
    x_right = []
    x_left = []
    x = []
    for x_values in correct.keys():
        x.append((x_values))
    for x_values in incorrect.keys():
        x_right.append(x_values+0.2)
    for x_values in total.keys():
        x_left.append(x_values-0.2)
    plt.figure(figsize=(10,6))
    plt.bar(x_right, list(incorrect.values()), 0.2, label='Incorrect prediction')
    plt.bar(x, list(correct.values()), 0.2, label='Correct prediction')
    plt.bar(x_left, list(total.values()), 0.2, label='Total in data')
    plt.title('Predictions with setup: ' + str(setup) + ' order: ' + str(order) + ',in total ' + str(perc) + '% correct')
    plt.xlabel("Word class")
    plt.ylabel("Predicted word classes")
    plt.xticks(x)
    plt.legend()
    plt.show()


def plot_confusion_metric_bar(accuracy, precision, recall,rows, setup):
    x = []
    x_right = []
    x_left = []
    number_wc = []
    counter=0
    for i in rows:
        number_wc.append(counter)
        counter +=1
    for x_values in number_wc:
        x_right.append((x_values + 0.2))
        x_left.append(x_values - 0.2)
        x.append(x_values)
    plt.figure(figsize=(16, 6))
    if not recall:
        plt.bar(x_right, accuracy, 0.2, label='F1-score')
        plt.bar(x, precision, 0.2, label='Specificity')
        plt.title("Metrics from confusion matrix with " + str(setup) + " and order " + str(len(setup) - 1))
        plt.ylabel("F1-score, Specificity")
    else:
        plt.bar(x_right, accuracy, 0.2, label='F1 score')
        plt.bar(x_left, precision, 0.2, label='Precision')
        plt.bar(x, recall, 0.2, label='Recall')
        plt.title("Metrics from confusion matrix with " + str(setup) + " and order " + str(len(setup)-1))
        plt.ylabel("F1 score, precision and recall")

    plt.xlabel("Word classes")
    plt.xticks(x)
    plt.legend()
    plt.show()

def plot_weights(weights, percentage_list, setup, nletters):
    #Getting weight with max to draw the line
    xmax = 0
    ymax = 0
    for i in range(len(percentage_list)):
        percentage_list[i] *= 100
        if percentage_list[i]> ymax:
            ymax=percentage_list[i]
            xmax=weights[i]

    plt.figure()
    plt.plot(weights, percentage_list, marker='o')
    plt.axvline(xmax, color='r', linestyle='--')
    plt.xlabel('Weight', fontsize=15)
    plt.ylabel('Right predictions out of total %', fontsize=15)
    plt.title('Setup = ' + str(setup) + ' Order ' + str(len(setup)-1) + ', ' + str(nletters) +' letters')
    plt.grid()
    plt.show()


def plot_fourier1(xf, yf, n):
    N = len(yf)
    # sample spacing
    plt.semilogy(xf, np.abs(yf[0:N//2]))
    plt.semilogy(xf, np.abs(yf[0:N//2]))
    plt.grid()
    plt.legend(['Non Translated', 'Translated full'])
    plt.xlabel('frequency')
    plt.ylabel('signal')
    plt.title('fourier transform for ' + str(n) + ' abstracts')
    plt.show()

def plot_fourier(xf, yf, n):
    N = len(yf[0])
    # sample spacing
    plt.semilogy(xf[0], np.abs(yf[0][0:N//2]))
    plt.semilogy(xf[1], np.abs(yf[1][0:N//2]))
    plt.grid()
    plt.legend(['Non Translated', 'Translated full'])
    plt.xlabel('frequency')
    plt.ylabel('signal')
    plt.title('fourier transform for ' + str(n) + ' abstracts')
    plt.show()
if __name__ == "__main__":
    pass


