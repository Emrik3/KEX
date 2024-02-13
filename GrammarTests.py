from metrics import *
from dataProcessing import open_dict, read_translation_txt, translations_to_word_classes, text_cleaner
from metrics import probofhappening2d, probofhappening1d, probofhappening3d, grammar_predictor, grammar_predictor2


def testinggrammar1d():
    text = read_translation_txt('Trainingdata/translated_sample.txt')
    classlist = translations_to_word_classes('Trainingdata/translated_sample.txt', "wordclasslists/WC_transl.json")
    TM_all = open_dict('transition_matrices/TM_all')
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



def testinggrammar2d():
    text = read_translation_txt('Trainingdata/translated_sample.txt')
    classlist = translations_to_word_classes('Trainingdata/translated_sample.txt', "wordclasslists/WC_transl.json")
    TM_all = open_dict('transition_matrices/TM_all_2nd')
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


def testinggrammar3d():
    text = read_translation_txt('Trainingdata/translated_sample.txt')
    classlist = translations_to_word_classes('Trainingdata/translated_sample.txt', "wordclasslists/WC_transl.json")
    TM_all = open_dict('transition_matrices/TM_all_3rd')
    p, error = probofhappening3d(TM_all, classlist)
    # print(p)
    # print(error)
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


def predict(file, giventext, WCgiventext, order):
    text = read_translation_txt(giventext)
    text = text_cleaner(text)
    sentences = text.lower().split('. ')
    textlist = []
    for sentence in sentences:
        words = sentence.split(' ')
        textlist.append(words)
    classlist = translations_to_word_classes(giventext, WCgiventext)
    TM = open_dict(file)
    if order == 1:
        res = grammar_predictor(TM, classlist, textlist)
    elif order == 2:
        res = grammar_predictor2(TM, classlist, textlist)
    elif order == 3:
        res = grammar_predictor3(TM, classlist, textlist)
        
    print(res)
    return res
