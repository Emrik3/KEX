# Basic graphs and so on
import matplotlib.pyplot as plt
import json
from TransitionMatrix import class_to_index
from collections import Counter
import numpy as np
from choose_word_classes import number_to_class
import pandas as pd
import seaborn as sns
from dataProcessing import save_dict
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
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
    ax.tick_params(axis='both', which='minor', labelsize=8)"""

def conf_mat_vis(matrix, total, k):
    """Generates heat map of transition matrix """
    newmat = np.zeros((len(matrix), len(matrix)))
    print(total)
    for i in list(total.keys()):
        for j in list(total.keys()):
            newmat[i][j] = matrix[i][j] / total[i]
    plt.rcParams["font.family"] = "georgia"
    
    fig, ax = plt.subplots(figsize=(20,20))
    k=0
    
    mat_size = max(class_to_index.values()) + 1
    ax.set_xticks(range(1, mat_size))
    ax.set_yticks(range(mat_size-2, -1, -1))
    keys_to_include = list(class_to_index.keys())[1:mat_size]
    ax.set_xticklabels(keys_to_include, rotation=45, fontsize=20)
    ax.set_yticklabels(keys_to_include, fontsize=20)
    ax.tick_params(axis='both', which='major') # Set fontsize on labels, this is hard to get right cause dont fit.
    ax.tick_params(axis='both', which='minor')
    im = ax.imshow(np.flipud(newmat), cmap='viridis', interpolation='nearest')
    if k == 1:
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size='5%', pad=0.6, pack_start = True)
        fig.add_axes(cax)
        
        cbar = fig.colorbar(im, orientation = 'horizontal', cax=cax)
        cbar.set_ticklabels(['{:,.0%}'.format(x/100) for x in range(0, 101, 20)], fontsize=20)
    plt.savefig('kexbilder/confmatrixord1letter3w03.pdf', bbox_inches='tight', format = 'pdf')
    k += 1
    plt.show()


def transition_matrix_vis(matrix):
    """Generates heat map of transition matrix """
    df = pd.DataFrame(matrix)
    plt.rcParams["font.family"] = "georgia"

    x = range(df.shape[1])
    y = range(df.shape[0])

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(15,15))
    # Plot the bubbles
    plt.grid(linestyle='--', color='gray')
    ax.scatter(X, Y, s=df.values*1000, color='black')  # Multiply by 1000 to make bubbles more visible
    plt.xticks(range(df.shape[1]), range(df.shape[1]))
    plt.yticks(range(df.shape[0]), range(df.shape[0]))
    mat_size = max(class_to_index.values()) + 1
    keys_to_include = list(class_to_index.keys())[0:mat_size]
    ax.set_xticklabels(keys_to_include, rotation=45, fontsize=20)
    ax.set_yticklabels(keys_to_include, fontsize=20)
    plt.xlabel('Next Word Class', fontsize=30)
    plt.ylabel('Current Word Class', fontsize=30)    
    plt.title('Confusion Matrix', fontsize=30)
    plt.savefig('kexbilder/TMvis.pdf', bbox_inches='tight', format = 'pdf')
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
    

def organize_and_plot(res, order, setup, plot, nletters, k):
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
    tot_incorrect = sum(list(total_occurrences.values())) #????? TODO

    #print("non_guess: "+ str(non_guess) + ", tot_correct: " + str(tot_correct) + ", total_total: " + str(tot_tot) + ", incorrect: " + str(tot_incorrect))
    if plot:
        print("Right predictions to incorrect ratio, total: " + (str(tot_correct/tot_incorrect)))
        print("Right predictions out of all made predictions: " + str((tot_correct)/(tot_tot-non_guess)))
        #confusion_metrics(confusionmatrix,setup)
        conf_mat_vis(confusionmatrix, total_occurrences, k)


    save_dict('results/plotdatapredict_correct_countsl' + str(nletters) + str(setup) + '.json', correct_counts)
    save_dict('results/plotdatapredict_wrong_countsl' + str(nletters) + str(setup) + '.json', wrong_counts)
    save_dict('results/plotdatapredict_total_occurrencesl' + str(nletters) + str(setup) + '.json', total_occurrences)
    #np.save('results/plotdatapredict_confusion_matrix' + str(setup) + '.npy', confusionmatrix)
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
def getF1(res):
    """Calculates 5 metrics using the confusion matrix"""
    confusionmatrix = np.zeros((len(class_to_index), len(class_to_index)))
    #print("res: " + str(len(res)))
    for elem in res:
        confusionmatrix += elem[3]
    data = 0
    matrix = confusionmatrix
    for i in range(len(matrix)):  # for some row
        FN = [0] * len(class_to_index)
        TP = [0] * len(class_to_index)
        FP = [0] * len(class_to_index)
        TN = [0] * len(class_to_index)
        counter = 0
        f1_0counter=0
        for j in range(len(matrix)):  # for element in row
            if i == j:
                TP[i] += matrix[i][j]  # diagonal elements give TP
            else:
                FN[i] += matrix[i][j]  # sums every other element in the row
                FP[i] += matrix[j][i]  # sums every other element in the column
            TN[i] = np.sum(matrix) - (TP[i] + FN[i] + FP[i])  # sum of all other values are true negatives
        if (TP[i] + FP[i]) > 0 and TP[i] > 0:  # divide by zero check
            Precision = round((TP[i]) / (TP[i] + FP[i]), 3)
            # print("Precision for word class " + str((number_to_class[i])) + "  : " + str(Precision*100) + "%")
            counter+=1
        else:
            Precision = 0.0
        if (TN[i] + FP[i]) > 0 and TP[i] > 0:  # divide by zero check
            Recall = round((TP[i]) / (TP[i] + FN[i]), 3)
            # print("Recall for word class " + str((number_to_class[i])) + "  : " + str(Recall*100))
        else:
            Recall = 0.0
        if counter == 1:  # divide by zero check
            F1_score = round((2 * Precision * Recall) / (Precision + Recall), 3)
            # print("F1 score for word class " + str((number_to_class[i])) + "  : " + str(100 * F1_score) + "%")
        else:
            F1_score = 0.0
            f1_0counter += 1
        data += F1_score
    print("sum of F1: " + str(data/len(matrix)))
    return data/len(matrix)

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


def plot_all_missed_beside(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2):
    x = []
    xi = []
    xv = []
    for x_values in correct.keys():
        x.append((int(x_values)-0.2))
    for x_values in incorrect.keys():
        xi.append((x_values))
    for x_values in total.keys():
        xv.append((int(x_values)+0.2))
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    colors = ['black', 'silver', 'navy', 'indigo', 'plum', 'purple', 'red', 'green', 'yellow', 'lavenderblush', 'greenyellow', 'lightgreen', 'chocolate', 'salmon']
    color_dict = {}
    for i in range(len(setuplist)):
        color_dict[str(setuplist[i])] = colors[i]
    print(correct.keys())
    #plt.plot(xi, list(incorrect.values()), label='Incorrect prediction for ' + str(setup))
    ax.bar(x, list(correct.values()), 0.2, label='Correct prediction for ' + str(setup), color=color_dict[str(setup)], bottom=bottom) #Maybe do percent instead, or do total at least, 
    ax.bar(xv, list(total.values()), 0.2, label='Total number of times predicted?? for ' + str(setup), color=color_dict[str(setup)], bottom=bottom2)
    bottom += np.array(list(correct.values()))
    bottom2 += np.array(list(total.values()))
    return bottom, bottom2
    #plt.plot(xv, list(total.values()), label='Total in data for ' + str(setup))


def plot_all_missed(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2):
    x = []
    xi = []
    xv = []
    print(correct)
    for x_values in correct.keys():
        x.append((int(x_values)-0.2))
    for x_values in incorrect.keys():
        xi.append((x_values))
    for x_values in total.keys():
        xv.append((int(x_values)+0.2))
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    colors = ['darksalmon', 'darkkhaki', 'darksalmon', 'darkkhaki', 'orange', 'darksalmon', 'darkkhaki', 'orange', 'moccasin', 'darksalmon', 'darkkhaki', 'orange', 'moccasin', 'goldenrod']
    color_dict = {}
    for i in range(len(setuplist)):
        color_dict[str(setuplist[i])] = colors[i]
    #plt.plot(xi, list(incorrect.values()), label='Incorrect prediction for ' + str(setup))
    ax.bar(x, list(correct.values()), 0.2, label='Correct prediction for ' + str(setup), color=color_dict[str(setup)], bottom=bottom, edgecolor='black') #Maybe do percent instead, or do total at least, 
    ax.bar(xv, list(total.values()), 0.2, label='Total number of tries to predict for ' + str(setup), color=color_dict[str(setup)], bottom=bottom2, hatch='//', edgecolor='black')
    bottom += np.array(list(correct.values()))
    bottom2 += np.array(list(total.values()))
    return bottom, bottom2
    #plt.plot(xv, list(total.values()), label='Total in data for ' + str(setup))


def plot_all_missed_subfigs(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2, k):
    x = []
    xi = []
    xv = []
    print(4,1,k)
    plt.subplot(4,1,k)
    for x_values in correct.keys():
        x.append((int(x_values)-0.2))
    for x_values in incorrect.keys():
        xi.append((x_values))
    for x_values in total.keys():
        xv.append((int(x_values)+0.2))
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    colors = ['darksalmon', 'darkkhaki', 'darksalmon', 'darkkhaki', 'orange', 'darksalmon', 'darkkhaki', 'orange', 'moccasin', 'darksalmon', 'darkkhaki', 'orange', 'moccasin', 'goldenrod']
    color_dict = {}
    for i in range(len(setuplist)):
        color_dict[str(setuplist[i])] = colors[i]
    print(incorrect)
    colorbest = {'[0, 1]': 'darksalmon', '[0, 1, 0]': 'darkkhaki', '[0, 0, 1, 0]': 'orange', '[0, 0, 0, 1, 0]': 'moccasin'}
    plt.bar(xi, list(incorrect.values()), 0.2, label='Inorrect prediction for ' + str(setup), color=colorbest[str(setup)], edgecolor='black')
    #plt.bar(x, list(correct.values()), 0.2, label='Correct prediction for ' + str(setup), color=colorbest[str(setup)], edgecolor='black') #Maybe do percent instead, or do total at least, 
    #plt.bar(xv, list(total.values()), 0.2, label='Total number of tries to predict for ' + str(setup), color=colorbest[str(setup)], hatch='//', edgecolor='black')
    #bottom += np.array(list(correct.values()))
    #bottom2 += np.array(list(total.values()))
    if k == 1:
        plt.title('Incorrect Predictions', fontsize=40)
        
    if k == 4:
        plt.xticks(range(1,len(class_to_index.keys())), list(class_to_index.keys())[1:], fontsize=25, rotation=45)
    else:
        plt.xticks(range(1,len(class_to_index.keys())), list(class_to_index.keys())[1:], fontsize=0)
    k += 1
    return bottom, bottom2, k
    #plt.plot(xv, list(total.values()), label='Total in data for ' + str(setup))


def plot_all_missed_bubble(corr_perc, correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2):
    x = []
    xi = []
    xv = []
    for x_values in correct.keys():
        x.append((int(x_values)))
    for x_values in incorrect.keys():
        xi.append((x_values))
    for x_values in total.keys():
        xv.append((int(x_values)))
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    colors = ['black', 'silver', 'navy', 'indigo', 'plum', 'purple', 'red', 'green', 'yellow', 'lavenderblush', 'greenyellow', 'lightgreen', 'chocolate', 'salmon']
    color_dict = {}
    for i in range(len(setuplist)):
        color_dict[str(setuplist[i])] = colors[i]
    #plt.plot(xi, list(incorrect.values()), label='Incorrect prediction for ' + str(setup))
    X, Y = np.meshgrid(x, list(correct.values()))
    ax.scatter(x, list(correct.values()), s=np.array(list(corr_perc.values()))*1000, label='Correct prediction for ' + str(setup), color=color_dict[str(setup)]) #Maybe do percent instead, or do total at least, 
    #ax.bar(xv, list(total.values()), 0.2, label='Total number of times predicted?? for ' + str(setup), color=color_dict[str(setup)], bottom=bottom2)
    bottom += np.array(list(correct.values()))
    bottom2 += np.array(list(total.values()))
    return bottom, bottom2
    #plt.plot(xv, list(total.values()), label='Total in data for ' + str(setup))
    

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

def plot_weights(weights, percentage_list, setup, nletters, convex):
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
    if convex:
        plt.title('Setup = ' + str(setup) + ' Order ' + str(len(setup) - 1) + ', ' + str(nletters) + ' letters, using:P_wc+P_letter ')
    else:
        plt.title('Setup = ' + str(setup) + ' Order ' + str(len(setup)-1) + ', ' + str(nletters) +' letters, using:P_wc*P_letter')
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


def plot_F1(F1_list, letter_list):
    print(F1_list)
    x = []
    pure_F1_list = []
    counter=0
    setups = []
    print(letter_list)
    for elem in (F1_list):
        pure_F1_list.append(elem[0])
        x.append(counter)
        #setups.append(str(elem[1]) + str(', l='+str(letter_list[counter])))
        setups.append(str(elem[1]))

        counter+=1
    fig = plt.figure()
    ax = fig.add_subplot(111)

    save_dict('results/f1predict3lettermanysetup.json', F1_list)
    save_dict('results/xpredict3lettermanysetup.json', x)
    plt.plot(x, pure_F1_list)
    plt.xlabel('Setup')
    ax.set_xticks(range(counter))
    ax.set_xticklabels(setups, rotation=45)
    plt.ylabel('F1 Score')
    plt.title('F1 score for different orders and setup, letter' + str(letter_list[0]))
    plt.legend()
    plt.grid()
    plt.show()


def plot_freq(WClist):
    # DO this for bible as well.
    count = pd.Series(WClist).value_counts()
    plt.rcParams["font.family"] = "georgia"
    fig, ax = plt.subplots(figsize=(25, 15))
    plt.style.use('seaborn-v0_8-whitegrid') # Find best style and use for all plots.
    
    sns.set_style("whitegrid")
    blue, = sns.color_palette("muted", 1)
    ax.plot(count, color=blue) # Could also be semilogy here.
    ax.fill_between(count.index, 0, count.values, alpha=.3)
    plt.ylabel('Number of Occurences', fontsize=40)
    plt.xlabel('Word Class',  fontsize=40)
    plt.title('Number of Occurences of Word Classes in Training Data',  fontsize=40)
    plt.grid(linestyle='--', color='gray')
    mat_size = max(class_to_index.values()) + 1
    keys_to_include = list(class_to_index.keys())[0:mat_size]
    plt.xticks(count.index, count.index, fontsize=25)
    plt.yticks(range(0,210000,25000), range(0,210000,25000), fontsize=25)
    plt.ylim(0,200000)
    plt.xlim('NN', 'MID')
    ax.set_xticklabels(count.index, rotation=45, fontsize=25)
    #ax.set_yticklabels(fontsize=20)
    plt.savefig('kexbilder/freq.pdf', bbox_inches='tight', format = 'pdf')
    plt.show()



if __name__ == "__main__":
    pass


