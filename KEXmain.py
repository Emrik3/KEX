from metrics import *
from Visualisation import *
from dataProcessing import open_dict
from GrammarTests import testinggrammar1d, testinggrammar2d, testinggrammar3d, predict

# Opening transition matrices
TM_all = open_dict('TM_all')
TM_transl = open_dict('TM_transl.json')
TM_non_transl = open_dict('TM_non_transl.json')
TM_all_2nd = open_dict('TM_all_2nd')




def main():
    """Uses the finished model to extract results"""

    """Calculates the basic metrics"""
    #running_metrics(TM_all, TM_transl)
    #running_metrics(TM_all, TM_non_transl)

    """Plots a 2D transition Matrix"""
    #transition_matrix_vis(TM_all)
    #transition_matrix_vis(np.subtract(TM_all,TM_transl))
    #transition_matrix_vis(np.subtract(TM_all,TM_non_transl))

    """Finds the most grammatically likely and all grammatically "impossible" sentences"""
    #testinggrammar1d()
    #testinggrammar2d()
    #testinggrammar3d()

    """Predicts the unknown words in a given text"""
    #predict('TM_all', 'real_sample.txt', 'WC_transl.json', 1)
    #predict('TM_all_2nd', 'translated_sample.txt', 'WC_transl.json', 2)
    #predict('TM_all_3rd', 'translated_sample.txt', 'WC_transl.json', 3)




if __name__ == '__main__':
    main()