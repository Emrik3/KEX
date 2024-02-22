from metrics import *
from dataProcessing import open_dict, read_translation_txt, translations_to_word_classes, text_cleaner, read_traning_csv
from choose_word_classes import number_to_class
import random as rnd
import copy


def testinggrammar1d(text_to_test, WC_text_to_test, TM_all):
    text = read_translation_txt(text_to_test)
    classlist = translations_to_word_classes(text_to_test, WC_text_to_test, no_NA=False)
    p, error = probofhappening1d(TM_all, classlist)
    # print(p)
    # print(error)
    wlist = []
    tlist = text.split('.')
    for s in tlist:
        wlist.append(s.split())
    for er in error:
        print("Zero probability of this happening: " + str(tlist[er[0]]))
        print("This is because of the sequence: " + str(str(wlist[er[0]][er[1]-2:er[1]])))
    pmax = 0
    imax = -1
    for i in range(len(p)-1):
        if p[i] > pmax:
            pmax = p[i]
            imax = i
    if imax == -1:
        print("Everything zero")
    else:
        print("The most \'normal\' sentence: " + str(tlist[imax]))
    text_with_prob = []
    for i in range(len(wlist)-1):
        text_with_prob.append((wlist[i],p[i]))

    return text_with_prob


def testinggrammar2d(text_to_test, WC_text_to_test, TM_all):
    text = read_translation_txt(text_to_test)
    classlist = translations_to_word_classes(text_to_test, WC_text_to_test, no_NA=False)
    p, error = probofhappening2d(TM_all, classlist)
    # print(p)
    # print(error)
    wlist = []
    tlist = text.split('.')
    for s in tlist:
        wlist.append(s.split())
    for er in error:
        print("Zero probability of this happening: " + str(tlist[er[0]]))
        print("This is because of the sequence: " + str(str(wlist[er[0]][er[1]-3:er[1]])))
    pmax = 0
    imax = -1
    for i in range(len(p)-1):
        if p[i] > pmax:
            pmax = p[i]
            imax = i
    if imax == -1:
        print("Everything zero")
    else:
        print("The most \'normal\' sentance: " + str(tlist[imax]))


def testinggrammar3d(text_to_test, WC_text_to_test, TM_all):
    text = read_translation_txt(text_to_test)
    classlist = translations_to_word_classes(text_to_test, WC_text_to_test, no_NA=False)
    p, error = probofhappening3d(TM_all, classlist)
    print(p)
    print(error)
    wlist = []
    tlist = text.split('.')
    for s in tlist:
        wlist.append(s.split())
    for er in error:
        print("Zero probability of this happening: " + str(tlist[er[0]]))
        print("This is because of the sequence: " + str(str(wlist[er[0]][er[1]-4:er[1]])))
    pmax = 0
    imax = -1
    for i in range(len(p)-1):
        if p[i] > pmax:
            pmax = p[i]
            imax = i
    if imax == -1:
        print("Everything zero")
    else:
        print("The most \'normal\' sentance: " + str(tlist[imax]))


def predict(TM, giventext, WC_list, orderfunc):
    # What happends when NA is given, that is when it should predict, fix this code for this but in other functin
    text = read_translation_txt(giventext)
    text = text_cleaner(text, no_dot=False)
    sentences = text.lower().split('. ')
    textlist = []
    for sentence in sentences:
        words = sentence.split(' ')
        textlist.append(words)
    res = orderfunc(TM, WC_list, textlist)
    return res

def predict_csv(TM, giventext, WC_list, orderfunc):
    # What happends when NA is given, that is when it should predict, fix this code for this but in other functin
    text = read_traning_csv(giventext)
    res = []
    for textt in text:
        textt = text_cleaner(textt, no_dot=False)
        sentences = textt.lower().split('. ')
        textlist = []
        for sentence in sentences:
            words = sentence.split(' ')
            textlist.append(words)
        res.append(orderfunc(TM, WC_list, textlist))
    return res

def probofhappening1d(A, classtext):
    # Kolla sannolikheten av grejer att komma efter varandra här!..
    classtextnum = []
    error = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    p = np.ones(len(result))
    for i in range(len(result)):
        for j in range(1, len(result[i])):
            p[i] *= A[result[i][j]][result[i][j - 1]]
            if A[result[i][j]][result[i][j - 1]] == 0:
                error.append((i, j))
    print(p)
    print(error)
    return p, error


def probofhappening2d(A, classtext):
    # Kolla sannolikheten av grejer att komma efter varandra här!..
    classtextnum = []
    error = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])

    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:

            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    p = np.ones(len(result))
    for i in range(len(result)):
        for j in range(2, len(result[i])):
            p[i] *= A[result[i][j]][result[i][j - 1]][result[i][j - 2]]
            if A[result[i][j]][result[i][j - 1]][result[i][j - 2]] == 0:
                error.append((i, j))
    print(p)
    return p, error


def probofhappening3d(A, classtext):
    # Kolla sannolikheten av grejer att komma efter varandra här!..
    classtextnum = []
    error = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])

    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:

            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)

    p = np.ones(len(result))
    for i in range(len(result)):
        for j in range(3, len(result[i])):
            p[i] *= A[result[i][j]][result[i][j - 1]][result[i][j - 2]][result[i][j - 3]]
            if A[result[i][j]][result[i][j - 1]][result[i][j - 2]][result[i][j - 3]] == 0:
                error.append((i, j))
    return p, error


def grammar_predictor(A, classtext, textlist):
    d = {}
    classtextnum = []
    error = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)

    maxprob = np.zeros(len(A))
    for i in range(len(A)):
        maxprob[i] = A[i].index(max(A[i]))
    for i in range(len(result)):
        for j in range(1, len(result[i]) - 2):
            if result[i][j] == 0:
                result[i][j] = maxprob[int(result[i][j - 1])]
                print(textlist[i][j] + " predicted as " + str(number_to_class[result[i][j]]))
                d[textlist[i][j]] = number_to_class[result[i][j]]
    return d


def grammar_predictor_percentage_test(A, classtext, textlist):
    """Does the same thing as grammar predictor but creates is own NA:s and ignores
    spots where NA exists. The old result is saved and compared to the prediction."""
    classtextnum = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    copy_result = copy.deepcopy(result)
    tot_counter = 0
    for i in range(len(result)):
        for j in range(1, len(result[i])-2):
            if rnd.randint(1, 10) == 10:  # every 1 out of 10 words
                if result[i][j] != 0 and result[i][j -1] != 0:  # current and following words are not already NA
                    result[i][j] = -1  # sets this word to NA
                    tot_counter += 1
    maxprob = np.zeros(len(A))
    for i in range(len(A)):
        maxprob[i] = A[i].index(max(A[i]))
    correct_counter = 0
    correct_predicted_class = []
    wrong_predicted_class = []
    wrong_actual_class = []
    for i in range(len(result)):
        for j in range(1, len(result[i]) - 2):
            if result[i][j] == -1: # creates -1 which doesn't exist in class to index and treats this as NA(0) was treated before
                result[i][j] = maxprob[int(result[i][j - 1])]
                if result[i][j] == copy_result[i][j]: #If the prediction was correct
                    correct_counter += 1
                    correct_predicted_class.append(copy_result[i][j]) #Doesn't matter if copy or not since they are the same
                else:
                    wrong_predicted_class.append(result[i][j])
                    wrong_actual_class.append(copy_result[i][j])
    #print("amount of words classfied correctly: " + str(correct_counter) + "of " + str(tot_counter))
    #print("in percent: " + str(100 * correct_counter / tot_counter) + "%")
    return wrong_predicted_class, wrong_actual_class, correct_predicted_class

def grammar_predictor2(A, classtext, textlist):
    classtextnum = []
    error = []
    d = {}
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)

    maxprob = np.zeros((len(A)-1, len(A)-1))
    for i in range(1,len(A)):
        for j in range(1,len(A)):
            maxprob[i-1][j-1] = A[i][j].index(max(A[i][j]))
    print(maxprob)
    for i in range(len(result)):
        for j in range(2, len(result[i]) - 1):
            if result[i][j] == 0:
                if result[i][j - 1] != '':
                    result[i][j] = maxprob[int(result[i][j - 2])][int(result[i][j - 1])]
                    if result[i][j] == 0:
                        result[i][j] = 1
                    print(textlist[i][j] + " predicted as " + str(number_to_class[result[i][j]]))
                    d[textlist[i][j]] = number_to_class[result[i][j]]
    return d

def grammar_predictor_percentage_test_2(A, classtext, textlist):
    """Does the same thing as grammar predictor but creates is own NA:s and ignores
    spots where NA exists. The old result is saved and compared to the prediction."""
    classtextnum = []
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    copy_result = copy.deepcopy(result)
    tot_counter = 0
    for i in range(len(result)):
        for j in range(1, len(result[i])-2):
            if rnd.randint(1, 10) == 10:  # every 1 out of 10 words
                if result[i][j] != 0 and result[i][j -1] != 0:  # current and following words are not already NA
                    result[i][j] = -1  # sets this word to NA
                    tot_counter += 1
    maxprob = np.zeros((len(A)-1, len(A)-1))
    for i in range(1,len(A)):
        for j in range(1,len(A)):
            maxprob[i-1][j-1] = A[i][j].index(max(A[i][j]))
    correct_counter = 0
    correct_predicted_class = []
    wrong_predicted_class = []
    wrong_actual_class = []
    for i in range(len(result)):
        for j in range(1, len(result[i]) - 2):
            if result[i][j] == -1: # creates -1 which doesn't exist in class to index and treats this as NA(0) was treated before
                result[i][j] = maxprob[int(result[i][j - 1][j-2])]
                if result[i][j] == copy_result[i][j]: #If the prediction was correct
                    correct_counter += 1
                    correct_predicted_class.append(copy_result[i][j]) #Doesn't matter if copy or not since they are the same
                else:
                    wrong_predicted_class.append(result[i][j])
                    wrong_actual_class.append(copy_result[i][j])
    #print("amount of words classfied correctly: " + str(correct_counter) + "of " + str(tot_counter))
    #print("in percent: " + str(100 * correct_counter / tot_counter) + "%")
    return wrong_predicted_class, wrong_actual_class, correct_predicted_class


def grammar_predictor3(A, classtext, textlist):
    classtextnum = []
    error = []
    d = {}
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)

    maxprob = np.zeros((len(A)-1, len(A)-1, len(A)-1))
    for i in range(1,len(A)):
        for j in range(1,len(A)):
            for k in range(1,len(A)):
                maxprob[i-1][j-1][k-1] = A[i][j][k].index(max(A[i][j][k]))
    print(maxprob)
    for i in range(len(result)):
        for j in range(3, len(result[i]) - 1):
            if result[i][j] == 0:
                if result[i][j - 1] != '':
                    result[i][j] = maxprob[int(result[i][j - 3])][int(result[i][j - 2])][int(result[i][j-1])]
                    if result[i][j] == 0:
                        result[i][j] = 1
                    print(textlist[i][j] + " predicted as " + str(number_to_class[result[i][j]]))
                    d[textlist[i][j]] = number_to_class[result[i][j]]
    return d


def grammar_predictor4(A, classtext, textlist):
    classtextnum = []
    error = []
    d = {}
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    maxprob = np.zeros((len(A), len(A), len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            for k in range(len(A)):
                for p in range(len(A)):
                    maxprob[i][j][k][p] = A[i][j][k][p].index(max(A[i][j][k][p]))
    for i in range(len(result)-1):
        for j in range(2, len(result[i]) - 4):
            if result[i][j] == 0:
                if result[i][j - 1] != '':
                    result[i][j] = maxprob[int(result[i][j - 4])][int(result[i][j - 3])][int(result[i][j-2])][int(result[i][j-1])]
                    if result[i][j] == 0:
                        result[i][j] = 1
                    print(textlist[i][j] + " predicted as " + str(number_to_class[int(result[i][j])]))
                    d[textlist[i][j]] = number_to_class[result[i][j]]
    return d


def grammar_predictor5(A, classtext, textlist):
    classtextnum = []
    error = []
    d = {}
    for i in range(len(classtext)):
        classtextnum.append(class_to_index[classtext[i]])
    particular_value = class_to_index['.']
    result = []
    temp_list = []
    for i in classtextnum:
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)

    maxprob = np.zeros((len(A), len(A), len(A), len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            for k in range(len(A)):
                for p in range(len(A)):
                    for q in range(len(A)):
                        maxprob[i][j][k][p][q] = A[i][j][k][p][q].index(max(A[i][j][k][p][q]))
    for i in range(len(result)):
        for j in range(5, len(result[i]) - 1):
            if result[i][j] == 0:
                if result[i][j - 1] != '':
                    result[i][j] = maxprob[int(result[i][j - 1])][int(result[i][j - 2])][int(result[i][j-3])][int(result[i][j-4])][int(result[i][j-5])]
                    print(result[i][j])
                    print(textlist[i][j] + " predicted as " + str(number_to_class[int(result[i][j])]))
                    d[textlist[i][j]] = number_to_class[result[i][j]]
    return d

