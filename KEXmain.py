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
from fourier import *



def update_TM(setup):
    """Updates Markov chains"""
    order = len(setup)-1
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
    #abstracts_to_word_classes(t1990_dir, WC_1990_dir, no_NA=False, segment=True, transl=False) 

    """Translates txt file to word classes"""
    #translations_to_word_classes(real_sample_dir, WC_non_transl_dir, no_NA= False)
    #translations_to_word_classes(translated_sample_dir, WC_transl_dir, no_NA = False)
    translations_to_word_classes('dictionaries/fixedbible.json', 'dictionaries/fixedbible.json', no_NA=False)

    """WC, with different lists for each abstract"""
    #abstracts_to_word_classes(export_dir, WC_export_segment_dir, no_NA=False, segment=True)
    #abstracts_to_word_classes(export_dir, WC_export_segment_fulltransl_dir, no_NA=False, segment=True, transl='Full')
    #abstracts_to_word_classes(export_dir, wc_export_segment_swtransl_dir, no_NA=False, segment=True, transl='Single')


def plot():
    """Plots a 2D transition Matrix"""
    transition_matrix_vis(np.load(TM_all_dir))
    #transition_matrix_vis(TM_transl)
    #transition_matrix_vis(TM_non_transl)

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


def metrics_test_translation(setup, type, n):
    order = len(setup)-1
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
        metric_fun = running_metrics2
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
    WC = open_dict(type)
    for i in range(len(WC_export_segment)):
        if prog<n: # Set to 17 if using single word translation, otherwise anything over 54
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


def metrics_test_scramble(setup):
    order = len(setup)-1
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
    n=5
    prog=0
    if order == 1:
        func = run_1_order
        metric_fun = running_metrics2
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
                func(copy_abs, TM_org_dir ,setup, mixed=False)
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
                if normal[7] < transl[7]:
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

def predict_big(setup, nletters, weights, plot, convex, F1_test):
    """Plots the results of predicting words in export_dir for some order of Markov chain"""
    order = len(setup)-1
    percentage_list = []
    for weight in (weights):
        print("weight" + str(weight))
        results = [grammar_predictor_main(WC_all, "export_dir", setup, order=order, nletters=nletters, weight=weight, convex=convex)]
        if not F1_test:
            percentage = organize_and_plot(results, order=order, setup=setup, plot=plot)
            percentage_list.append(percentage)
    if F1_test:
        return getF1(results)
    else:
        plot_weights(weights, percentage_list, setup, nletters, convex)

def predict_big_data(setup, nletters, weights, plot, convex, F1_test):
    """Plots the results of predicting words in export_dir for some order of Markov chain"""
    
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    for setup in setuplist:
        update_TM(setup=setup)
        order = len(setup)-1
        for weight in (weights):
            print("setup: " + str(setup))
            results = [grammar_predictor_main(WC_all, "export_dir", setup, order=order, nletters=nletters, weight=weight, convex=convex)]
            percentage = organize_and_plot(results, order=order, setup=setup, plot=plot)


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
    #prob_ending_class(textlist, wclist)
    #prob_ending2_class(textlist, wclist)
    prob_ending3_class(textlist, wclist)


def predict_ending():
    textlist = open_dict('Trainingdata/abstracts_textlist')
    wclist = open_dict('wordclasslists/WC_all.json')
    A = open_dict('transition_matrices/TM_all.json')
    predictor_with_endings1(A, wclist, textlist, [0,1])

def get_url():
    # get URL
    page = requests.get("https://sv.wikipedia.org/wiki/Lista_%C3%B6ver_sj%C3%A4lvst%C3%A4ndiga_stater")
    soup = BeautifulSoup(page.content, 'html.parser')
    # display scraped data
    print(soup.prettify())

def fourier_run():
    xf, yf, n = fourier_test(np.load(TM_all_dir), WC_export_segment)
    xf1, yf1, m = fourier_test(np.load(TM_all_dir), WC_export_segment_fulltransl)
    print(n, m)
    plot_fourier([xf, xf1], [yf, yf1], n)

def test_fourier_no_compare():
    xf, yf, n = fourier_test_no_smooth(np.load(TM_all_dir), WC_all)
    print(n)
    plot_fourier1(xf, yf, n)
    

def predict_many_F1():
    nletters_list = [3] #[-1, 0, 1, 2]
    F1_list = []
    letter_list = []
    for nletters in nletters_list:
        org_letter = nletters
        weight =[0.5]
        if nletters ==0:
            weight = [1]
            nletters=1
        if nletters == -1:
            weight = [1]
            nletters=1
        setup = [0,1]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights = weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)
        
        setup = [1,0]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights = weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

        setup = [0, 0, 1]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights=weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

        setup = [0, 1,0]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights=weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

        setup = [0,0,1,0]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights=weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

        setup = [0, 0, 1,0, 0]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights=weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

        setup = [0, 0, 0, 1, 0]
        update_TM(setup=setup)
        F1_list.append((predict_big(setup=setup, nletters=nletters, weights=weight, plot=False, convex=False, F1_test=True), setup))
        letter_list.append(org_letter)

    plot_F1(F1_list, letter_list)

def test_pearson():
    xf, Y1, n = fourier_test(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y2, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    xf, Y3, n = fourier_test_for_1990(np.load(TM_all_dir), open_dict(WC_1990_dir))
    xf, Y4, n = fourier_test(np.load(TM_all_dir), WC_all_segment)
    xf, Y5, n = fouriertest_shuffla(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y6, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    print("PEARSON")
    print()
    print("First and second half of all abstracts compared:")
    print((pearson_corr_coeff(np.abs(Y1), np.abs(Y2))))
    print()
    print("All abstracts compared with full 1990 file")
    print((pearson_corr_coeff(np.abs(Y3), np.abs(Y4))))
    print()
    print("Shuffled")
    print((pearson_corr_coeff(np.abs(Y5), np.abs(Y6))))


def test_spearman():
    # THe number of words used in the fourier test functions differ a lot, this should be fixed by cheking what happends.
    xf, Y1, n = fourier_test(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y2, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    xf, Y3, n = fourier_test_for_1990(np.load(TM_all_dir), open_dict(WC_1990_dir))
    xf, Y4, n = fourier_test(np.load(TM_all_dir), WC_all_segment)
    xf, Y5, n = fouriertest_shuffla(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y6, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    print("SPEARMAN")
    print()
    print("First and second half of all abstracts compared:")
    print((spearman_corr_coeff(np.abs(Y1), np.abs(Y2))))
    print()
    print("All abstracts compared with full 1990 file")
    print((spearman_corr_coeff(np.abs(Y3), np.abs(Y4))))
    print()
    print("Shuffled")
    print((spearman_corr_coeff(np.abs(Y5), np.abs(Y6))))


def test_sam():
    # No work
    # THe number of words used in the fourier test functions differ a lot, this should be fixed by cheking what happends.
    xf, Y1, n = fourier_test(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y2, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    xf, Y3, n = fourier_test_for_1990(np.load(TM_all_dir), open_dict(WC_1990_dir))
    xf, Y4, n = fourier_test(np.load(TM_all_dir), WC_all_segment)
    xf, Y5, n = fouriertest_shuffla(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y6, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    print("SPEARMAN")
    print()
    print("First and second half of all abstracts compared:")
    print((spec_ang_map(np.abs(Y1), np.abs(Y2))))
    print()
    print("All abstracts compared with full 1990 file")
    print((spec_ang_map(np.abs(Y3), np.abs(Y4))))
    print()
    print("Shuffled")
    print((spec_ang_map(np.abs(Y5), np.abs(Y6))))


def test_dist_corr():
    # Very bad
    # THe number of words used in the fourier test functions differ a lot, this should be fixed by cheking what happends.
    xf, Y1, n = fourier_test(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y2, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    xf, Y3, n = fourier_test_for_1990(np.load(TM_all_dir), open_dict(WC_1990_dir))
    xf, Y4, n = fourier_test(np.load(TM_all_dir), WC_all_segment)
    xf, Y5, n = fouriertest_shuffla(np.load(TM_all_dir), WC_all_segment[0:len(WC_all_segment)//2])
    xf, Y6, n = fourier_test(np.load(TM_all_dir), WC_all_segment[len(WC_all_segment)//2:])
    print("Distance Corr")
    print()
    print("First and second half of all abstracts compared:")
    print((dist_corr(np.abs(Y1), np.abs(Y2))))
    print()
    print("All abstracts compared with full 1990 file")
    print((dist_corr(np.abs(Y3), np.abs(Y4))))
    print()
    print("Shuffled")
    print((dist_corr(np.abs(Y5), np.abs(Y6))))


def test_any4(corr, corr2, corr3, corr4):
    # Anything
    # Need more text the p-value is too large! 
    """More text maybe, p-value very large for shuffled kendall tau..."""
    # THe number of words used in the fourier test functions differ a lot, this should be fixed by cheking what happends.
    xf, Y01, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(WC_all)[0:len(WC_all)//2])
    xf, Y02, n = fourier_test_for_bible(np.load(TM_all_dir), WC_all[len(WC_all)//2:])


    xf, Y1, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))[0:len(open_dict(bible_WC_dir))//2])
    xf, Y2, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))[len(open_dict(bible_WC_dir))//2:])

    xf, Y3, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))) # Many NA in the bible, stop at val is to have as many avrages as abstracts, has to do with p-value, idk why...
    xf, Y4, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(WC_all))

    xf, Y5, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))) # This gives very different values, sometimes they are very large... Prob not working as it should
    xf, Y6, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))

    xf, Y9, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(WC_all)) # This gives very different values, sometimes they are very large... Prob not working as it should
    xf, Y10, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))

    xf, Y7, n = fourier_test_for_1990(np.load(TM_all_dir), copy.deepcopy(open_dict(WC_1990_dir)))
    xf, Y8, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))
    
    print("First and second half of abstracts compared:")
    print((corr(np.abs(Y01), np.abs(Y02))[0] + corr2(np.abs(Y01), np.abs(Y02))[0] + corr3(np.abs(Y01), np.abs(Y02))[0]) / corr4(np.abs(Y01), np.abs(Y02))[0])
    print()
    print("First and second half of bible compared:")
    print(((corr(np.abs(Y1), np.abs(Y2))[0])+(corr2(np.abs(Y1), np.abs(Y2))[0]) + corr3(np.abs(Y1), np.abs(Y2))[0]) / corr4(np.abs(Y1), np.abs(Y2))[0])
    print()
    print("All abstracts compared with the bible")
    print((corr(np.abs(Y3), np.abs(Y4))[0]+corr2(np.abs(Y3), np.abs(Y4))[0] + corr3(np.abs(Y3), np.abs(Y4))[0]) / corr4(np.abs(Y3), np.abs(Y4))[0])
    print()
    print("Shuffled bible compared with bible")
    print(((corr(np.abs(Y5), np.abs(Y6))[0])+(corr2(np.abs(Y5), np.abs(Y6))[0]) + corr3(np.abs(Y5), np.abs(Y6))[0]) / corr4(np.abs(Y5), np.abs(Y6))[0])
    print()
    print("Shuffled abstracts compared with bible")
    print(((corr(np.abs(Y9), np.abs(Y10))[0])+(corr2(np.abs(Y9), np.abs(Y10))[0]) + corr3(np.abs(Y9), np.abs(Y10))[0]) / corr4(np.abs(Y9), np.abs(Y10))[0])
    print()
    print("1990 and bible")
    print(((corr(np.abs(Y7), np.abs(Y8))[0])+(corr2(np.abs(Y7), np.abs(Y8))[0]) + corr3(np.abs(Y7), np.abs(Y8))[0]) / corr4(np.abs(Y7), np.abs(Y8))[0])


def test_any(corr, corr2):
    # Anything
    # Need more text the p-value is too large! 
    """More text maybe, p-value very large for shuffled kendall tau..."""
    # THe number of words used in the fourier test functions differ a lot, this should be fixed by cheking what happends.
    xf, Y01, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(WC_all)[0:len(WC_all)//2])
    xf, Y02, n = fourier_test_for_bible(np.load(TM_all_dir), WC_all[len(WC_all)//2:])

    xf, Y1, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))[0:len(open_dict(bible_WC_dir))//2])
    xf, Y2, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))[len(open_dict(bible_WC_dir))//2:])

    xf, Y3, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))) # Many NA in the bible, stop at val is to have as many avrages as abstracts, has to do with p-value, idk why...
    xf, Y4, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(WC_all))

    xf, Y5, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))) # This gives very different values, sometimes they are very large... Prob not working as it should
    xf, Y6, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))

    xf, Y9, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(WC_all)) # This gives very different values, sometimes they are very large... Prob not working as it should
    xf, Y10, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))

    xf, Y7, n = fourier_test_for_1990(np.load(TM_all_dir), copy.deepcopy(open_dict(WC_1990_dir)))
    xf, Y8, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))
    print(str(corr))
    print()
    print("First and second half of abstracts compared:")
    print((corr(np.abs(Y01), np.abs(Y02))[0]))
    print()
    print("First and second half of bible compared:")
    print((corr(np.abs(Y1), np.abs(Y2))[0]))
    print()
    print("All abstracts compared with the bible")
    print((corr(np.abs(Y3), np.abs(Y4))[0]))
    print()
    print("Shuffled bible compared with bible")
    print((corr(np.abs(Y5), np.abs(Y6))[0]))
    print()
    print("Shuffled abstracts compared with bible")
    print((corr(np.abs(Y9), np.abs(Y10))[0]))
    print()
    print("1990 and bible")
    print((corr(np.abs(Y7), np.abs(Y8))[0]))


def shuffle_avg(corr, corr2):
    avg_bib = []
    avg_abs = []
    for i in range(100):
        print(str(i) + "%")
        xf, Y9, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(WC_all)) # This gives very different values, sometimes they are very large... Prob not working as it should
        xf, Y10, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))
        avg_abs.append((corr(np.abs(Y9), np.abs(Y10))[0])+(corr2(np.abs(Y9), np.abs(Y10))[0]))

        xf, Y5, n = fourier_test_shuffle_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir))) # This gives very different values, sometimes they are very large... Prob not working as it should
        xf, Y6, n = fourier_test_for_bible(np.load(TM_all_dir), copy.deepcopy(open_dict(bible_WC_dir)))
        avg_bib.append((corr(np.abs(Y5), np.abs(Y6))[0])+(corr2(np.abs(Y9), np.abs(Y10))[0]))

    print("Avrage for abstracts")
    print(sum(avg_abs)/100)
    print()
    print("Avrage for bible")
    print(sum(avg_bib)/100)



def plot_all_subfigs(): 
    count = pd.Series(open_dict(WC_all_dir)).value_counts()
    k = 1
    plt.rcParams["font.family"] = "georgia"
    fig, ax = plt.subplots(figsize=(25,20))
    
    setuplist1 = [[0,1], [1,0]]
    setuplist2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    setuplist3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] 
    setuplist4 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    setuplistbest = [[0,1], [0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1, 0]]
    bottom = np.zeros(len(class_to_index.keys()))
    bottom2 = np.zeros(len(class_to_index.keys()))
    for setup in setuplistbest:
        print(setup)
        ordr = len(setup)-1
        correct = open_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json')
        incorrect = open_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json')
        correct_perc = {}
        total = open_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json')
        
        for key in total.keys():
            try:
                correct_perc[key] = correct[key] / total[key]
            except:
                correct_perc[key] = 0
        bottom, bottom2, k = plot_all_missed_subfigs(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2, k)
        
        
        
        plt.yticks(fontsize=25)
        #plt.set_xticklabels(list(class_to_index.keys())[1:], fontsize=20, rotation=45) # Probably wrong, NA should not be there.
        #ax.set_yscale('log') # This for bar log plot
        plt.grid(linestyle='--', color='gray')
        plt.legend(prop={'size': 25})
        plt.xlim(0.5, 25)
    
    
    plt.xlabel("Word class", fontsize=40)
    #plt.ylabel("Predicted word classes", fontsize=40)
    plt.savefig('kexbilder/predbestsubincorr.pdf', bbox_inches='tight', format = 'pdf')
    plt.show()

def plot_all():
    count = pd.Series(open_dict(WC_all_dir)).value_counts()
    plt.rcParams["font.family"] = "georgia"
    fig, ax = plt.subplots(figsize=(25,15))
    k = 1
    setuplist1 = [[0,1], [1,0]]
    setuplist2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    setuplist3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] 
    setuplist4 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    bottom = np.zeros(len(class_to_index.keys()))
    bottom2 = np.zeros(len(class_to_index.keys()))
    plt.title('Predictions with setup', fontsize=40)
    for setup in setuplist4:
        ordr = len(setup)-1
        correct = open_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json')
        incorrect = open_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json')
        correct_perc = {}
        total = open_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json')
        
        for key in total.keys():
            try:
                correct_perc[key] = correct[key] / total[key]
            except:
                correct_perc[key] = 0
        bottom, bottom2= plot_all_missed(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2)
    plt.xlabel("Word class", fontsize=40)
    plt.ylabel("Predicted word classes", fontsize=40)
    plt.xticks(range(1,len(class_to_index.keys())), list(class_to_index.keys())[1:], rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    #plt.set_xticklabels(list(class_to_index.keys())[1:], fontsize=20, rotation=45) # Probably wrong, NA should not be there.
    #ax.set_yscale('log') # This for bar log plot
    plt.grid(linestyle='--', color='gray')
    plt.legend(prop={'size': 25})
    
    plt.savefig('kexbilder/pred5stack.pdf', bbox_inches='tight', format = 'pdf')
    
    plt.show()


def all_percent_no_letter():
    plt.rcParams["font.family"] = "georgia"
    fig, ax = plt.subplots(figsize=(25,15))
    setuplist = [[0,1], [1,0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    bottom = np.zeros(len(class_to_index.keys()))
    bottom2 = np.zeros(len(class_to_index.keys()))
    plt.title('Correct Predictions', fontsize=40)
    blue, = sns.color_palette("muted", 1)
    perc = {}
    for setup in setuplist:
        ordr = len(setup)-1
        correct = open_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json')
        incorrect = open_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json')
        correct_perc = {}
        total = open_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json')

        tot_correct = sum(correct.values())
        tot = sum(total.values())
        perc[str(setup)] = round(tot_correct/tot * 100, 2)

    keys = list(perc.keys())
    values = list(perc.values())
    sorted_value_index = np.argsort(values)
    perc = {keys[i]: values[i] for i in sorted_value_index}
    #plt.style.use('seaborn-v0_8-whitegrid') # Find best style and use for all plots.
    
    
    ax.plot(perc.keys(), perc.values(), color=blue, marker='D', markersize=7, markeredgecolor='black') # Could also be semilogy here.
    ax.fill_between(perc.keys(), 0, perc.values(), alpha=.3)
    plt.ylabel('Percent Correct', fontsize=40)
    plt.xlabel('Setup',  fontsize=40)
    plt.grid('--', color='gray')
    mat_size = max(class_to_index.values()) + 1
    keys_to_include = list(class_to_index.keys())[0:mat_size]
    plt.xticks(list(perc.keys()), list(perc.values()), fontsize=25)
    #plt.yticks(range(0,210000,25000), range(0,210000,25000), fontsize=20)
    plt.ylim(0,50)
    plt.xlim('[1, 0]', '[0, 0, 0, 1, 0]')
    ax.set_xticklabels(list(perc.keys()), rotation=45, fontsize=25)
    ax.set_yticklabels(['{:,.2%}'.format(x/100) for x in range(0, 51, 10)], fontsize=25)
    plt.savefig('kexbilder/allpercnoletter.pdf', bbox_inches='tight', format = 'pdf')
    plt.show()

        #print(str(setup) + ': ' + str(round(tot_correct/tot * 100, 2)) + '%')


def plot_all_subfigs_weights():
    count = pd.Series(open_dict(WC_all_dir)).value_counts()
    k = 1
    fig, ax = plt.subplots(figsize=(10,10))
    setuplist1 = [[0,1], [1,0]]
    setuplist2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    setuplist3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] 
    setuplist4 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    bottom = np.zeros(len(class_to_index.keys()))
    bottom2 = np.zeros(len(class_to_index.keys()))
    plt.title('Predictions with setup', fontsize=25)
    for setup in setuplist4:
        ordr = len(setup)-1
        correct = open_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json')
        incorrect = open_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json')
        correct_perc = {}
        total = open_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json')
        
        for key in total.keys():
            try:
                correct_perc[key] = correct[key] / total[key]
            except:
                correct_perc[key] = 0
        bottom, bottom2, k = plot_all_missed_subfigs(correct, incorrect, total, ordr, setup, ax, count, bottom, bottom2, k)
        plt.xlabel("Word class", fontsize=20)
        plt.ylabel("Predicted word classes", fontsize=20)
        plt.xticks(range(1,len(class_to_index.keys())), list(class_to_index.keys())[1:], rotation=45, fontsize=15)
        plt.yticks(fontsize=20)
        #plt.set_xticklabels(list(class_to_index.keys())[1:], fontsize=20, rotation=45) # Probably wrong, NA should not be there.
        #ax.set_yscale('log') # This for bar log plot
        plt.grid(linestyle='--', color='black')
        plt.legend()
    
    
    plt.show()

def fix_data_plot():
    setuplist = [[0, 1], [1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    sz = len(class_to_index.keys())
    for setup in setuplist:
        ordr = len(setup)-1
        correct = open_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json')
        incorrect = open_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json')
        total = open_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json')
        for cl in range(len(class_to_index.keys())):
            try:
                a = correct[str(cl)]
            except:
                correct[str(cl)] = 0
            try:
                a = incorrect[str(cl)]
            except:
                incorrect[str(cl)] = 0
            try:
                a = total[str(cl)]
            except:
                total[str(cl)] = 0
        save_dict('results/plotdatapredict_correct_counts' + str(setup) + '.json', correct)
        save_dict('results/plotdatapredict_wrong_counts' + str(setup) + '.json', incorrect)
        save_dict('results/plotdatapredict_total_occurrences' + str(setup) + '.json', total)

def main():
    """Uses the finished model to extract results"""
    #update_WC()
    #setup = [0, 1]
    #update_TM(setup=setup)
    #plot()
    #metrics()
    #evaluate_grammar()
    #predict_NA()
    #update_end_prob()
    #create_end_wc_matrix()
    #get_url()
    #create_end_wc_matrix()
    # BElow just to look at the matrix and what is non zero, only ones and zeros, dont know why, look at this...
    #predict_ending()
    #m = np.load('wordclasslists/WCending.npy')
    setup = [0,1,0]
    #update_TM(setup=setup)
    """1. Predict Word Classes"""
    #predict_big(setup=setup, nletters=3, weights = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1], plot=False) # Look at when is does not identify wht is it equal to then, i mean when it skips due to words like i and so on.
    #predict_big(setup=setup, nletters=1, weights = [1], plot=False, convex=False, F1_test=False) # Samma som nletters=0
    #predict_big(setup=setup, nletters=2, weights = [0.5], plot=True) # Samma som utan weight
    #predict_big(setup=setup, nletters=1, weights = [0, 0.01, 0.3, 0.4,0.5,0.6, 0.99, 1], plot=False, convex=False, F1_test=False) # Look at when is does not identify wht is it equal to then, i mean when it skips due to words like i and so on.
    #predict_many_F1()
    #predict_big(setup=setup, nletters=1, weights = [1], plot=True, convex=False, F1_test=False) # Samma som nletters=0
    #predict_big(setup=setup, nletters=2, weights = [0.5], plot=True, convex=True, F1_test=False) # Samma som utan weight
    #predict_big_data(setup=None, nletters=1, weights = [1], plot=False, convex=False, F1_test=False)

    """2. Testing the grammar of translation software"""
    #metrics_test_translation(setup=setup, type=WC_export_segment_fulltransl_dir, n=100) # Remember to update_TM() if using a new setup
    #metrics_test_translation(setup=setup, type=wc_export_segment_swtransl_dir, n=17) # Remember to update_TM() if using a new setup
    #metrics_test_scramble(setup=setup)

    """3. Fourier transform to find patterns in text (to be further implemented)"""
    #fourier_run()
    #test_fourier_no_compare()
    #test_pearson()
    #test_spearman()
    #test_sam() # No work
    #test_dist_corr()
    #update_WC()
    #fix_data_plot()
    #plot_all()
    #plot_all_subfigs()
    #all_percent_no_letter()
    #print(len(WC_all))
    #print(len(open_dict(WC_export_dir)))
    #print(len(open_dict(WC_1990_dir)))
    #print(len(open_dict('dictionaries/fixedbible.json')))


    # List of functions: use scipy.stats. before: pearsonr, spearmanr (Depends a lot on n), pointbiserialr, kendalltau, weightedtau, somersd, siegelslopes, theilslopes
    # Best: kendalltau, weightedtau (Hyperbolic weighing)
    # Bad: somersd
    test_any4(scipy.stats.spearmanr, scipy.stats.spearmanr, scipy.stats.pearsonr, spec_ang_map)
    #test_any(spec_ang_map, 1)
    #shuffle_avg(scipy.stats.spearmanr, scipy.stats.kendalltau) # Very large number of stuff from the bible, idk why.
    # So left to do: Kendal tau and weighted need to run the shuffle multiple times and avrage it, also do convergence prots for that and make tabel of the values...
    #plot_freq(WC_all)

    # TODO: Är allt rätt innan vi gör plottarna!!!!!
    # TODO: Find a plt.style.use('seaborn-v0_8-whitegrid') and use it for all plots.
    # TODO: make grph like 6.2.3 in joars text.


if __name__ == '__main__':
    main()