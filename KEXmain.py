from metrics import *
from Visualisation import *
from dataProcessing import open_dict
from GrammarTests import testinggrammar1d, testinggrammar2d, testinggrammar3d, predict, grammar_predictor, grammar_predictor2, grammar_predictor3, grammar_predictor4, grammar_predictor5
from TransitionMatrix import *

# Opening transition matrices
TM_all = open_dict('transition_matrices/TM_all')
TM_transl = open_dict('transition_matrices/TM_transl.json')
TM_non_transl = open_dict('transition_matrices/TM_non_transl.json')
TM_all_2nd = open_dict('transition_matrices/TM_all_2nd')
TM_transl_2nd = open_dict('transition_matrices/TM_transl_2nd.json')
TM_non_transl_2nd = open_dict('transition_matrices/TM_non_transl_2nd.json')

import requests
from bs4 import BeautifulSoup
 


def main():
    """Uses the finished model to extract results"""
    """Updates Markov chains"""
    #run_1_order('wordclasslists/WC_all.json', "transition_matrices/TM_all")
    #run_1_order('wordclasslists/WC_transl.json', "transition_matrices/TM_transl.json")
    #run_1_order('wordclasslists/WC_non_transl.json', 'transition_matrices/TM_non_transl.json')
    #run_5_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_5th')

    #run_2_order('wordclasslists/WC_transl.json', 'transition_matrices/TM_transl_2nd.json')
    #run_2_order('wordclasslists/WC_non_transl.json', 'transition_matrices/TM_non_transl_2nd.json')
    #run_4_order('wordclasslists/WC_all.json', 'transition_matrices/TM_all_4th')

    """Calculates the basic metrics"""
    """running_metrics(TM_all, TM_transl)
    running_metrics(TM_all, TM_non_transl)"""

    #running_metrics(TM_all_2nd, TM_transl_2nd)
    #running_metrics(TM_all_2nd, TM_non_transl_2nd)


    """Plots a 2D transition Matrix"""
    """transition_matrix_vis(TM_all)
    transition_matrix_vis(np.subtract(TM_all,TM_transl))
    transition_matrix_vis(np.subtract(TM_all,TM_non_transl))"""

    """Finds the most grammatically likely and all grammatically "impossible" sentences"""
    """testinggrammar1d('Trainingdata/translated_sample.txt', "wordclasslists/WC_transl.json",'transition_matrices/TM_all')
    testinggrammar2d('Trainingdata/translated_sample.txt', 'wordclasslists/WC_transl.json', 'transition_matrices/TM_all_2nd')
    testinggrammar3d('Trainingdata/real_sample.txt', 'wordclasslists/WC_non_transl.json', 'transition_matrices/TM_all_3rd')
    testinggrammar3d('Trainingdata/translated_sample.txt', 'wordclasslists/WC_transl.json', 'transition_matrices/TM_all_3rd')"""


    """Predicts the unknown words in a given text"""
    #predict('transition_matrices/TM_all', 'Trainingdata/real_sample.txt', 'wordclasslists/WC_transl.json', grammar_predictor)
    #predict('transition_matrices/TM_all_2nd', 'Trainingdata/translated_sample.txt', 'wordclasslists/WC_transl.json', grammar_predictor2)
    #predict('transition_matrices/TM_all_3rd', 'Trainingdata/translated_sample.txt', 'wordclasslists/WC_transl.json', grammar_predictor3)
    #predict('transition_matrices/TM_all_4th', 'Trainingdata/real_sample.txt', 'wordclasslists/WC_transl.json', grammar_predictor4)
    predict('transition_matrices/TM_all_4th', 'Trainingdata/real_sample.txt', 'wordclasslists/WC_transl.json', grammar_predictor4)
    # get URL
    """page = requests.get("https://sv.wikipedia.org/wiki/Lista_%C3%B6ver_sj%C3%A4lvst%C3%A4ndiga_stater")
    
    soup = BeautifulSoup(page.content, 'html.parser')
 
    # display scraped data
    print(soup.prettify())"""




if __name__ == '__main__':
    main()