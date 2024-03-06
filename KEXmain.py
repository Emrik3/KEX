from metrics import *
from Visualisation import *
from dataProcessing import *
from GrammarTests import *
from TransitionMatrix import *
import requests
from bs4 import BeautifulSoup
from choose_word_classes import *
from sympy import *



def update_TM():
    """Updates Markov chains"""
    run_1_order(WC_all_dir,TM_all_dir)
    #run_1_order(WC_transl_dir, TM_transl_dir)
    #run_1_order(WC_non_transl_dir, TM_non_transl_dir)

    #run_2_order(WC_all_dir, TM_all_2nd_dir)
    #run_3_order(WC_all_dir, TM_all_3rd_dir)
    #run_4_order(WC_all_dir, TM_all_4th_dir, setup = [0, 0, 1, 0, 0]) #1 för den man vill kolla på
    #run_5_order(WC_all_dir, TM_all_5th_dir, setup = [0, 0, 1, 0, 0, 0])
    #run_6_order(WC_all_dir, TM_all_6th_dir, setup = [0, 0, 0, 1, 0, 0, 0])

def update_WC():
    """Translates web-scraped csv files to word classes"""
    abstracts_to_word_classes(Training_data_dir,WC_all_dir, no_NA=False)
    abstracts_to_word_classes(export_dir, WC_export_dir, no_NA=False)

    """Translates txt file to word classes"""
    #translations_to_word_classes(real_sample_dir, WC_non_transl_dir, no_NA= False)
    #translations_to_word_classes(translated_sample_dir, WC_transl_dir, no_NA = False)
def plot():
    """Plots a 2D transition Matrix"""
    transition_matrix_vis(TM_all)
    #transition_matrix_vis(TM_transl)
    #transition_matrix_vis(TM_non_transl)

def metrics():
    """Calculates the basic metrics"""
    running_metrics(TM_all, TM_transl)
    running_metrics(TM_all, TM_non_transl)

def evaluate_grammar():
    """Finds the most grammatically likely and all grammatically "impossible" sentences"""
    ### Funkar inte för tillfället. Varje sekvens med NA ord sätts till 0 sannolikhet då NA inte finns i TM
    testinggrammar1d(real_sample_dir, WC_non_transl_dir, TM_all)
    testinggrammar2d(real_sample_dir, WC_non_transl_dir, TM_all_2nd)
    testinggrammar3d(real_sample_dir, WC_non_transl_dir, TM_all_3rd)

def predict_NA():
    """Predicts the unknown words in a given text"""

    """One thing that happends is if it is a row with all zeos the first index is returned aand gives NA because of that
    I fized this by replacing NA with substantiv in the printed result
    But this should not happen unless there is a ordföljd that never happend before!!!"""

    #predict(TM_all, translated_sample_dir, WC_transl, grammar_predictor)
    #predict(TM_all_2nd, translated_sample_dir, WC_transl, grammar_predictor2)
    #predict(TM_all_3rd, translated_sample_dir, WC_transl, grammar_predictor3)
    #predict((Matrix(TM_all)*(Matrix(TM_all_future).T)).tolist(), translated_sample_dir, WC_transl, grammar_predictor)


    #Using a csv file for a larger test
    #results = predict_csv(TM_all, export_dir, WC_export, grammar_predictor_percentage_test)
    #organize_and_plot(results)

    #results = predict_csv(TM_all_2nd, export_dir, WC_export, grammar_predictor_percentage_test2)
    #organize_and_plot(results)

    #results = predict_csv(TM_all_3rd, export_dir, WC_export, grammar_predictor_percentage_test3)
    #organize_and_plot(results)

    """Bör lägga till [0, 0, 1, 0, 0] som argument här men för tillfället funkar det såhär:
    1. Kör run_4_order med vald config t.ex[0, 0, 1,0, 0]
    2. Gå till GrammarTests ->grammar_predictor_percentage_test4 och 
    ändra j+/- (på 2 ställen) till det som överenstämmer med konfigurationen t.ex.[0, 1,0,0,0]-> j-1, j+1, j+2, j+3
    Ändra rangen på dessa 2 ställen också"""
    results = predict_csv(TM_all_4th, export_dir, WC_export, grammar_predictor_percentage_test4)
    organize_and_plot(results)

    #5th order now with .npy format
    #results = predict_csv(TM_all_5th, export_dir, WC_export, grammar_predictor_percentage_test5)
    #organize_and_plot(results)

    #6th order when TM_all_6th.npy exists, currently too slow normalization

def update_end_prob():
    text = read_traning_csv('Trainingdata/many_abstracts.csv')
    fulltext = ""
    for abstract in text:
        if check_english(abstract.split()):
            continue
        abstract = text_cleaner(abstract, no_dot=True)
        fulltext += ' ' + abstract
    ending_freq(fulltext, ending_list)


def get_url():
    # get URL
    page = requests.get("https://sv.wikipedia.org/wiki/Lista_%C3%B6ver_sj%C3%A4lvst%C3%A4ndiga_stater")

    soup = BeautifulSoup(page.content, 'html.parser')

    # display scraped data
    print(soup.prettify())

def main():
    """Uses the finished model to extract results"""
    #update_WC()
    #update_TM()
    plot()
    #metrics()
    #evaluate_grammar()
    #predict_NA()
    #update_end_prob()
    #get_url()


if __name__ == '__main__':
    main()