from dataProcessing import open_dict
import numpy as np
# File locations
TM_all_dir = 'transition_matrices/TM_all'
TM_transl_dir = 'transition_matrices/TM_transl.json'
TM_non_transl_dir = 'transition_matrices/TM_non_transl.json'

TM_all_2nd_dir = 'transition_matrices/TM_all_2nd'
TM_transl_2nd_dir = 'transition_matrices/TM_transl_2nd.json'
TM_non_transl_2nd_dir = 'transition_matrices/TM_non_transl_2nd.json'

TM_all_3rd_dir = 'transition_matrices/TM_all_3rd'
TM_all_4th_dir = 'transition_matrices/TM_all_4th'
TM_all_5th_dir = 'transition_matrices/TM_all_5th.npy'
TM_all_6th_dir = 'transition_matrices/TM_all_6th.npy'

TM_all_future_dir = 'transition_matrices/TM_all_future'


# Opening transition matrices
TM_all = open_dict(TM_all_dir)
TM_transl = open_dict(TM_transl_dir)
TM_non_transl = open_dict(TM_non_transl_dir)
TM_all_2nd = open_dict(TM_all_2nd_dir)
TM_transl_2nd = open_dict(TM_transl_2nd_dir)
TM_non_transl_2nd = open_dict(TM_non_transl_2nd_dir)
TM_all_3rd = open_dict(TM_all_3rd_dir)
TM_all_4th = open_dict(TM_all_4th_dir)
TM_all_5th = np.load(TM_all_5th_dir)
#TM_all_6th = np.load(TM_all_6th_dir)
TM_all_future = open_dict(TM_all_future_dir)

# Opening wordclasslists
WC_all_dir = "wordclasslists/WC_all.json"
WC_transl_dir = 'wordclasslists/WC_transl.json'
WC_non_transl_dir = 'wordclasslists/WC_non_transl.json'
WC_export_dir = 'wordclasslists/WC_export'
WC_export_segment_dir = 'wordclasslists/WC_export_segment'

# Opening wordclasslists
WC_all = open_dict(WC_all_dir)
WC_transl = open_dict(WC_transl_dir)
WC_non_transl = open_dict(WC_non_transl_dir)
WC_export = open_dict(WC_export_dir)
WC_export_segment = open_dict(WC_export_segment_dir)

# text to analyse
Training_data_dir = 'Trainingdata/many_abstracts.csv'
translated_sample_dir = 'Trainingdata/translated_sample.txt'
real_sample_dir = 'Trainingdata/real_sample.txt'
export_dir = 'Trainingdata/export.csv'


"""Choosing which word classes should be part of the transition matrix"""
class_to_index = {
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    'AB': 5,
    'PM': 6,
    '.': 7,
    'DT': 8,
    'KN': 9,
    'PC': 10,
    'SN': 11,
    'PS': 12,
    'PN': 13,
    'HP': 14,
    'RO': 15,
    'HA': 16,
    'PL': 17,
    'MAD': 18,
    'UO' : 19,
    'HD' : 20,
    'RG' : 21,
    'IN' : 22,
    'HS' : 23,
    'MID' : 24
    }
class_to_indexnoNA = {
    'NN': 0,
    'VB': 1,
    'PP': 2,
    'JJ': 3,
    'AB': 4,
    'PM': 5,
    '.': 6,
    'DT': 7,
    'KN': 8,
    'PC': 9,
    'SN': 10,
    'PS': 11,
    'PN': 12,
    'HP': 13,
    'RO': 14,
    'HA': 15,
    'PL': 16,
    'MAD': 17,
    'UO' : 18,
    'HD' : 19,
    'RG' : 20,
    'IN' : 21,
    'HS' : 22,
    'MID' : 23
    }
class_to_index_ww = {
    #Lite motivationer, kommer raffineras
    #PN = PM = PS = IN <=> pronomen = egennamn dvs Han = Stockholm = hans = AJ!
    #HA = HP <=> frågande pronomen = frågande adverb dvs vem = när
    #NA = UO (utländskt ord)
    # MAD = . = PAD = MID? dvs . = . = .,; (just nu iaf då vi filtrerar bort nästan allt)
    #  HS= något HS =vars, vems osv
    # PL = nåt finns bara en i datan, kollade i classdict och hittade exemplet "tillinitiativ" som en enskild sträng??
    #  RG= RO två = andra
    # HD (relativt besätmning) exemplet från classdict är "hurdana"??
    # SN subjunktion exemplet från classdict är "50som"??
    # IE  verkar vara tom
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    'AB': 5,
    'PN': 6,
    '.': 7,
    'DT': 8,
    'KN': 9,
    'PC': 10,
    'SN': 11,
    'HP': 12,
    'RO': 13,
    'PS': 6,
    'PM': 1,
    'HA': 12,
    'PAD': 7,
    'PL': 0,
    'MAD': 7,
    'UO' : 0,
    'HD' : 12,
    'RG' : 13,
    'MID' : 7,
    'IN' : 6,
    'HS' : 12
}

class_to_index_w = {
    # Pronomen = substantiv Han är där = Grejen är där
    # Adjektiv = Adverb Han är glad, Han springer fort
    # konjunktion = preposition ( gissar lite )
    # particip = adjektiv "Particip är ordformer som utgår från verb, men fungerar som adjektiv. "(källaisof)
    # Subjunktion = konjunktion (båda binder ihop satsdelar)
    # HP = pronomen (vem är där? = han är där)
    # RO = adjektiv (Han är först, han är på andra plats = Han är bäst, han är på sämsta platsen) lite oklart men kanske
    'NA': 0,
    'NN': 1,
    'VB': 2,
    'PP': 3,
    'JJ': 4,
    '.': 5,
    'DT': 6,
    'AB': 4,
    'PM': 1,
    'KN': 3,
    'PC': 4,
    'SN': 3,
    'HP': 1,
    'RO': 4,
    'PS': 1,
    'PN': 1,
    'HA': 1,
    'PAD': 5,# ev Problem
    'PL': 0,
    'MAD': 5, # ev Problem
    'UO' : 0,
    'HD' : 1,
    'RG' : 4,
    'MID' : 5, # ev problem
    'IN' : 1,
    'HS' : 1
}

number_to_class = {
    0: 'NA',
    1: 'Substantiv',
    2: 'Verb',
    3: 'Preposition',
    4: 'Adjektiv',
    5: 'Adverb',
    6: 'Preposition',
    7: 'Punkt',
    8: 'Determinerare',
    9: 'Konjunktion',
    10: 'Particip',
    11: 'Subjunktion',
    12: 'Possessivuttryck',
    13: 'Pronomen',
    14: 'Frågande/rel pronomen',
    15: 'Ordningstal',
    16: 'Frågande/rel adverb',
    17: 'Partikel',
    18: 'Meningsskiljande interpunktion',
    19: 'Utländskt ord',
    20: 'Frågande/relativt bestämning',
    21: 'Grundtal',
    22: 'Interjektion',
    23: 'Frågande/rel possesivuttryck',
    24: "Interpunktion (sista punkten)"
}
number_to_class_small = {
    0: "NA",
    1: "Substantiv, pronomen, egennamn etc",
    2: "Verb",
    3: "Preposition, konjunktion etc",
    4: "Adjektiv",
    5: "Punkt",
    6: "Determinerare"
}
def create_ending_list():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Å', 'Ä', 'Ö']
    ending_list = []
    for letter1 in letters:
        for letter2 in letters:
            ending_list.append(letter1.lower() + letter2.lower())

    return ending_list

ending_list = create_ending_list()



if __name__ == '__main__':
    print(create_ending_list())
