from metrics import *
from Visualisation import *
from dataProcessing import *
from GrammarTests import *
from TransitionMatrix import *
import requests
from bs4 import BeautifulSoup
from choose_word_classes import *
from sympy import *
import copy



def update_TM(order, setup):
    """Updates Markov chains"""
    if order == 1:
        func = run_1_order
        TM_dir = TM_all_dir
    elif order == 2:
        func = run_2_order
        TM_dir = TM_all_2nd_dir
    elif order == 3:
        func = run_3_order
        TM_dir = TM_all_3rd_dir
    elif order == 4:
        func = run_4_order
        TM_dir = TM_all_4th_dir
    elif order == 5:
        func = run_5_order
        TM_dir = TM_all_5th_dir
    else:
        print("ERROR: Choose the order of markov chain as 1,2,3,4 or 5")
        return
    func(WC_all_dir, TM_dir,setup, mixed=False)


def update_WC():
    """Translates web-scraped csv files to word classes"""
    #abstracts_to_word_classes(Training_data_dir,WC_all_dir, no_NA=False, segment=False, transl=False)


    """Translates txt file to word classes"""
    #translations_to_word_classes(real_sample_dir, WC_non_transl_dir, no_NA= False)
    #translations_to_word_classes(translated_sample_dir, WC_transl_dir, no_NA = False)

    """WC, with different lists for each abstract"""
    #abstracts_to_word_classes(export_dir, WC_export_segment_dir, no_NA=False, segment=True)
    #abstracts_to_word_classes(export_dir, WC_export_segment_fulltransl_dir, no_NA=False, segment=True, transl='Full')
    #abstracts_to_word_classes(export_dir, wc_export_segment_swtransl_dir, no_NA=False, segment=True, transl='Single')


def plot():
    """Plots a 2D transition Matrix"""
    #transition_matrix_vis(TM_all)
    #transition_matrix_vis(TM_transl)
    transition_matrix_vis(TM_non_transl)

def metrics():
    """Calculates the basic metrics"""
    #running_metrics(TM_all, TM_transl)
    #running_metrics(TM_all, TM_non_transl)
    running_metrics2(TM_all_2nd, TM_transl_2nd)
    running_metrics2(TM_all_2nd, TM_non_transl_2nd)

    print("Should be high")
    running_metrics2(TM_all_3rd, TM_transl_3rd)
    print("Should be low")
    running_metrics2(TM_all_3rd, TM_non_transl_3rd)
def metrics_test_translation(order, setup, type):
    # For running test using a translation
    prog = 0
    counter_1norm = 0
    counter_2norm = 0
    counter_infnorm = 0
    counter_frobnorm = 0
    counter_crossE = 0
    counter_kullback = 0
    counter_singularv = 0
    counter_wassenstein = 0
    normal = [] #metrics for normal version
    transl = [] #metrics for translated version
    other_metrics = [0]*6
    if order == 1:
        func = run_1_order
        metric_fun = running_metrics
        TM_big_dir = TM_all_dir
        TM_org_dir = TM_transl_non_dir
        TM_trnsl_dir = TM_transl_dir
    elif order == 2:
        func = run_2_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_2nd_dir
        TM_org_dir = TM_non_transl_2nd_dir
        TM_trnsl_dir = TM_transl_2nd_dir
    elif order == 3:
        func = run_3_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_3rd_dir
        TM_org_dir = TM_non_transl_3rd_dir
        TM_trnsl_dir = TM_transl_3rd_dir
    elif order == 4:
        func = run_4_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_4th_dir
        TM_org_dir = TM_non_transl_4th_dir
        TM_trnsl_dir = TM_transl_4th_dir
    elif order == 5:
        func = run_5_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_5th_dir
        TM_org_dir = TM_non_transl_5th_dir
        TM_trnsl_dir = TM_transl_5th_dir
    else:
        print("ERROR: Choose the order of markov chain as 1,2,3,4 or 5")
        return
    WC = open_dict(type)
    for i in range(len(WC_export_segment)):
        if prog<17: # Set to 17 if using single word translation, otherwise anything over 54
            abstract_org = WC_export_segment[i]
            abstract_tnsl = WC[i]  # full<->sw to change test
            prog +=1
            print("Progress: " + str(prog) + "/" + str(len(WC)))
            func(abstract_org, TM_org_dir, setup, mixed=False)
            normal.append(metric_fun(np.load(TM_big_dir), np.load(TM_org_dir)))
            func(abstract_tnsl, TM_trnsl_dir,setup, mixed=False)
            transl.append(metric_fun(np.load(TM_big_dir), np.load(TM_trnsl_dir)))
            other_metrics[0] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[0]
            other_metrics[1] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[1]
            other_metrics[2] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[2]
            other_metrics[3] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[3]
            other_metrics[4] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[4]
            other_metrics[5] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[5]

    length = prog
    for i in range(length):
        if normal[i] == transl[i]:
            length -= 1
            continue
        if normal[i][0] < transl[i][0]:
            counter_1norm +=1
        if normal[i][1] < transl[i][1]:
            counter_2norm += 1
        if normal[i][2] < transl[i][2]:
            counter_infnorm +=1
        if normal[i][3] < transl[i][3]:
            counter_frobnorm +=1
        if normal[i][4] < transl[i][4]:
            counter_crossE +=1
        if normal[i][5] < transl[i][5]:
            counter_kullback += 1
        if normal[i][6] < transl[i][6]:
            counter_singularv += 1
        if normal[i][7] < transl[i][7]:
            counter_wassenstein += 1

    print("Amount of abstracts which changed: " + str(length) + " out of " + str(prog))
    print("correct percentage 1-norm: " + str(counter_1norm/(length)*100) + "%")
    print("corrent percentage 2-norm: " + str(counter_2norm/(length)*100) + "%")
    print("correct percentage inf-norm: " + str(counter_infnorm/(length)*100) + "%")
    print("correct percentage Frob-norm: " + str(counter_frobnorm/(length)*100) + "%")
    print("correct percentage Cross entropy: " + str(counter_crossE/(length)*100) + "%")
    print("correct percentage Kullback: " + str(counter_kullback / (length) * 100) + "%")
    print("largest singular value: " + str(counter_singularv / (length) * 100) + "%")
    print("wassenstein: " + str(counter_wassenstein / (length) * 100) + "%")

    print("DIFFERENCE BETWEEN ORIGINAL AND TRANSLATED TEXT, MEASURES SEVERITY OF TRANSLATION/SHUFFLE")
    print("largest singular value: " + str(other_metrics[0] / (prog)))
    print("Wessenstein (cost): " + str(other_metrics[1] / (prog)))
    print("1-norm: " + str(other_metrics[2] / (prog)))
    print("Frob-norm: " + str(other_metrics[3] / (prog)))
    print("cross entropy: " + str(other_metrics[4] / (prog)))
    print("Kullback: " + str(other_metrics[5] / (prog)))

    print("Setup = " + str(setup))
def metrics_test_scramble(order, setup):
    #For running tests with only scrambled word order
    counter_1norm = 0
    counter_2norm = 0
    counter_infnorm = 0
    counter_frobnorm = 0
    counter_crossE = 0
    counter_kullback = 0
    counter_singularv = 0
    counter_wassenstein = 0
    other_metrics = [0]*6
    n=2
    prog=0
    if order == 1:
        func = run_1_order
        metric_fun = running_metrics
        TM_big_dir = TM_all_dir
        TM_org_dir = TM_non_transl_dir
        TM_trnsl_dir = TM_transl_dir
    elif order == 2:
        func = run_2_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_2nd_dir
        TM_org_dir = TM_non_transl_2nd_dir
        TM_trnsl_dir = TM_transl_2nd_dir
    elif order == 3:
        func = run_3_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_3rd_dir
        TM_org_dir = TM_non_transl_3rd_dir
        TM_trnsl_dir = TM_transl_3rd_dir
    elif order == 4:
        func = run_4_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_4th_dir
        TM_org_dir = TM_non_transl_4th_dir
        TM_trnsl_dir = TM_transl_4th_dir
    elif order == 5:
        func = run_5_order
        metric_fun = running_metrics2
        TM_big_dir = TM_all_5th_dir
        TM_org_dir = TM_non_transl_5th_dir
        TM_trnsl_dir = TM_transl_5th_dir
    else:
        print("ERROR: Choose the order of markov chain as 1,2,3,4 or 5")
        return
    for i in range(n): #Only relevant when there is randomness involved like with mixing word order
        print(i)
        prog = 0
        for abstract in WC_export_segment:
            if prog <100: #just to get the 13th in case this matrix should be plotted
                prog +=1
                copy_abs = copy.deepcopy(abstract) #use copy_abs for both and change mixed for the second to True if a scramble test
                # then use abstract org
                func(copy_abs, TM_non_transl_dir,setup, mixed=False)
                func(copy_abs, TM_trnsl_dir, setup, mixed=True)
                transl = metric_fun(np.load(TM_big_dir), np.load(TM_trnsl_dir))
                normal = metric_fun(np.load(TM_big_dir), np.load(TM_org_dir))
                other_metrics[0] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[0]
                other_metrics[1] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[1]
                other_metrics[2] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[2]
                other_metrics[3] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[3]
                other_metrics[4] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[4]
                other_metrics[5] += running_change_metrics(np.load(TM_trnsl_dir), np.load(TM_org_dir))[5]



                if normal[0] < transl[0]:
                    counter_1norm +=1
                if normal[1] < transl[1]:
                    counter_2norm +=1
                if normal[2] < transl[2]:
                    counter_infnorm +=1
                if normal[3] < transl[3]:
                    counter_frobnorm +=1
                if normal[4] < transl[4]:
                    counter_crossE +=1
                if normal[5] < transl[5]:
                    counter_kullback +=1
                if normal[6] < transl[6]:
                    counter_singularv +=1
                if normal[7] > transl[7]:
                    counter_wassenstein +=1

    print("correct percentage 1-norm: " + str(counter_1norm / (n*prog) * 100) + "%")
    print("corrent percentage 2-norm: " + str(counter_2norm / (n*prog) * 100) + "%")
    print("correct percentage inf-norm: " + str(counter_infnorm / (n*prog) * 100) + "%")
    print("correct percentage Frob-norm: " + str(counter_frobnorm / (n*prog) * 100) + "%")
    print("correct percentage Cross entropy: " + str(counter_crossE / (n*prog) * 100) + "%")
    print("correct percentage kullback: " + str(counter_kullback / (n*prog) * 100) + "%")
    print("correct percentage largest singular value: " + str(counter_singularv / (n * prog) * 100) + "%")
    print("correct percentage wassenstein: " + str(counter_wassenstein / (n * prog) * 100) + "%")

    print("DIFFERENCE BETWEEN ORIGINAL AND TRANSLATED TEXT, MEASURES SEVERITY OF TRANSLATION/SHUFFLE")
    print("largest singular value: " + str(other_metrics[0]/(n*prog)))
    print("Wessenstein (cost): " + str(other_metrics[1]/(n*prog)))
    print("1-norm: " + str(other_metrics[2] / (n*prog)))
    print("Frob-norm: " + str(other_metrics[3] / (n*prog)))
    print("cross entropy: " + str(other_metrics[4] / (n*prog)))
    print("Kullback: " + str(other_metrics[5] / (n*prog)))



    print("Setup = " + str(setup))



def evaluate_grammar():
    """Finds the most grammatically likely and all grammatically "impossible" sentences"""
    ### Funkar inte för tillfället. Varje sekvens med NA ord sätts till 0 sannolikhet då NA inte finns i TM
    #testinggrammar1d(real_sample_dir, WC_non_transl_dir, TM_all)
    #testinggrammar2d(real_sample_dir, WC_non_transl_dir, TM_all_2nd)
    #testinggrammar3d(real_sample_dir, WC_non_transl_dir, TM_all_3rd)
    #byt namn på nontransldir

def predict_NA():
    """Predicts the unknown words in a given text (vet inte om denna funkar längre)"""

    """One thing that happends is if it is a row with all zeos the first index is returned aand gives NA because of that
    I fixed this by replacing NA with substantiv in the printed result
    But this should not happen unless there is a ordföljd that never happend before!!!"""

    #predict(TM_all, translated_sample_dir, WC_transl, grammar_predictor)
    #predict(TM_all_2nd, translated_sample_dir, WC_transl, grammar_predictor2)
    #predict(TM_all_3rd, translated_sample_dir, WC_transl, grammar_predictor3)
    #predict((Matrix(TM_all)*(Matrix(TM_all_future).T)).tolist(), translated_sample_dir, WC_transl, grammar_predictor)

def predict_big(order, setup):
    """Plots the results of predicting words in export_dir for some order of Markov chain"""
    if order ==1:
        TM_dir = TM_all_dir
        grammar_pred_test = grammar_predictor_percentage_test
    elif order ==2:
        TM_dir = TM_all_2nd_dir
        grammar_pred_test = grammar_predictor_percentage_test2
    elif order == 3:
        TM_dir = TM_all_3rd_dir
        grammar_pred_test = grammar_predictor_percentage_test3
    elif order == 4:
        TM_dir = TM_all_4th_dir
        grammar_pred_test = grammar_predictor_percentage_test4
    elif order == 5:
        TM_dir = TM_all_5th_dir
        grammar_pred_test = grammar_predictor_percentage_test5
    else:
        print("ERROR: Choose the order of markov chain as 1,2,3,4 or 5")
        return
    results = predict_csv(np.load(TM_dir), export_dir, WC_export, grammar_pred_test, setup)
    organize_and_plot(results, order=order)

def update_end_prob():
    text = read_traning_csv('Trainingdata/many_abstracts.csv')
    fulltext = ""
    for abstract in text:
        if check_english(abstract.split()):
            continue
        abstract = text_cleaner(abstract, no_dot=True)
        fulltext += ' ' + abstract
    ending_freq(fulltext, ending_list)

def create_end_wc_matrix():
    textlist = open_dict('Trainingdata/abstracts_textlist')
    wclist = open_dict('wordclasslists/WC_all.json')
    prob_ending_class(textlist, wclist)


def predict_ending():
    textlist = open_dict('Trainingdata/abstracts_textlist')
    wclist = open_dict('wordclasslists/WC_all.json')
    A = open_dict('transition_matrices/TM_all.json')
    predictor_with_endings(A, wclist, textlist)

def get_url():
    # get URL
    page = requests.get("https://sv.wikipedia.org/wiki/Lista_%C3%B6ver_sj%C3%A4lvst%C3%A4ndiga_stater")
    soup = BeautifulSoup(page.content, 'html.parser')
    # display scraped data
    print(soup.prettify())

def main():
    """Uses the finished model to extract results"""
    #update_WC()
    #update_TM(order=4, setup=[0, 0, 1,0,0])
    #plot()
    #metrics()
    #evaluate_grammar()
    #predict_NA()
    #predict_big(order=4, setup=[0, 0, 1,0,0])
    #update_end_prob()
    #prob_ending_class(export_dir, 'dictionaries/classdict.json')
    #get_url()
    #metrics_test_translation(order=2, setup=[0, 1,0], type=WC_export_segment_fulltransl_dir) # Remember to update_TM() if using a new setup
    #metrics_test_translation(order=2, setup=[0, 1,0], type=wc_export_segment_swtransl_dir) # Remember to update_TM() if using a new setup
    #metrics_test_scramble(order=2, setup=[0, 1,0])
    #create_end_wc_matrix()
    # BElow just to look at the matrix and what is non zero, only ones and zeros, dont know why, look at this...
    predict_ending()
    
    



if __name__ == '__main__':
    main()