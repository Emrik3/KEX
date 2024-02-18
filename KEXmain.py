from metrics import *
from Visualisation import *
from dataProcessing import *
from GrammarTests import *
from TransitionMatrix import *
import requests
from bs4 import BeautifulSoup
from choose_word_classes import *



def update_TM():
    """Updates Markov chains"""
    run_1_order(WC_all_dir,TM_all_dir)
    run_1_order(WC_transl_dir, TM_transl_dir)
    run_1_order(WC_non_transl_dir, TM_non_transl_dir)

    run_2_order(WC_all_dir, TM_all_2nd_dir)
    run_3_order(WC_all_dir, TM_all_3rd_dir)
    run_4_order(WC_all_dir, TM_all_4th_dir, setup = [0, 0, 0, 0, 1]) #1 för den man vill kolla på
    #run_5_order(WC_all_dir, TM_all_5th_dir)

def update_WC():
    """Translates web-scraped csv files to word classes"""
    abstracts_to_word_classes(Training_data_dir, no_NA=True)

    """Translates txt file to word classes"""
    translations_to_word_classes(real_sample_dir, WC_non_transl_dir, no_NA= False)
    translations_to_word_classes(translated_sample_dir, WC_transl_dir, no_NA = False)

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
    predict(TM_all, translated_sample_dir, WC_transl, grammar_predictor)
    predict(TM_all_2nd, translated_sample_dir, WC_transl, grammar_predictor2)
    predict(TM_all_3rd, translated_sample_dir, WC_transl, grammar_predictor3)
    #predict(TM_all_4th,translated_sample_dir, WC_transl, grammar_predictor4)

def get_url():
    # get URL
    """page = requests.get("https://sv.wikipedia.org/wiki/Lista_%C3%B6ver_sj%C3%A4lvst%C3%A4ndiga_stater")

    soup = BeautifulSoup(page.content, 'html.parser')

    # display scraped data
    print(soup.prettify())"""

def main():
    """Uses the finished model to extract results"""
    #update_WC()
    #update_TM()
    plot()
    metrics()
    evaluate_grammar()
    predict_NA()
    #get_url()


if __name__ == '__main__':
    main()