from scipy.fft import fft
from choose_word_classes import class_to_index
import numpy as np


def fourier_test(WC):
    signal_list = []
    for _ in range(len(class_to_index)):
        signal_list.append([0] * len(WC)) # 1 signal list for each wc
    counter = 0
    for wc in WC:
        signal_list[class_to_index[wc]][counter] += 1 # adds a 1 where wc occurs
        counter += 1
    n=2 #word class to look at
    fourier = fft(signal_list[n]) #fourier transform of [0,1,0,0,0,1,0,0] where 1 is occurance of wordclass n
    frequency = np.fft.fftfreq(len(signal_list[n]), d=1)
    return frequency, fourier, n