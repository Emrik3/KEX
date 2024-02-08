import numpy as np
from metrics import *
from Visualisation import *
from dataProcessing import open_dict, read_translation_txt, translations_to_word_classes
from metrics import probofhappening


# first order
TM_all = open_dict('TM_all.json')
TM_transl = open_dict('TM_transl.json')
TM_non_transl = open_dict('TM_non_transl.json')

# second order
TM_all_2nd = open_dict('TM_all_2nd')

def testinggrammar():
    text = read_translation_txt('translated_sample.txt')
    classlist = translations_to_word_classes('translated_sample.txt', "WC_transl.json")
    TM_all = open_dict('TM_all.json')
    p, error = probofhappening(TM_all, classlist)
    print(p)
    print(error)
    wlist = []
    tlist = text.split('.')
    for s in tlist:
        wlist.append(s.split())
    for er in error:
        print("Zero probability of this happening: " + str(wlist[er[0]][er[1]-2:er[1]+2]))
    pmax = 0
    for i in range(len(p)-1):
        if p[i] > pmax:
            pmax = p[i]
            imax = i
    print("The most \'normal\' sentance: " + str(tlist[i]))

def main():
    """print("Using a non translated abstract")
    running_metrics(TM_all, TM_transl)
    print("Using a translated abstract:")
    running_metrics(TM_all, TM_non_transl)
    transition_matrix_vis(TM_all)"""
    testinggrammar()
    #transition_matrix_vis(np.subtract(TM_all,TM_transl))
    #transition_matrix_vis(np.subtract(TM_all,TM_non_transl))




if __name__ == '__main__':
    main()
