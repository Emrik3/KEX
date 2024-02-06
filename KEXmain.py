import numpy as np
from metrics import *
from Visualisation import *
from dataProcessing import open_dict, read_translation_txt

TM_all = open_dict('TM_all.json')
TM_transl = open_dict('TM_transl.json')
TM_non_transl = open_dict('TM_non_transl.json')

def main():
    print("Using a non translated abstract")
    running_metrics(TM_all, TM_transl)
    print("Using a translated abstract:")
    running_metrics(TM_all, TM_non_transl)
    transition_matrix_vis(TM_all)
    transition_matrix_vis(np.subtract(TM_all,TM_transl))
    transition_matrix_vis(np.subtract(TM_all,TM_non_transl))
    



if __name__ == '__main__':
    main()
