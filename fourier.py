from scipy.fft import fft, fftfreq
from choose_word_classes import class_to_index
import numpy as np


def fourier_test(A, WClist):
    n = 0
    yftot = []
    nwords = 100
    for WC in WClist:
        if len(WC) >= nwords:
            clist = cross_entropy_sequence(A, WC[0:nwords])
            #https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
            yf = fft(clist) 
            N = len(yf)
            # sample spacing
            T = 1.0 / 800.0
            xf = fftfreq(N, T)[:N//2]
            if n == 0:
                yftot = np.array(yf) # Take abs here or just in the end?
                n+=1
            elif not np.isinf(np.abs(yf[0])):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
    print(n)
    yftot = yftot * (1 / n)
    return xf, yftot, n

def fourier_test_no_smooth(A, WClist):
    n = 0
    yftot = []
    for i in range(100, int(len(WClist)/1000)):
        clist = cross_entropy_sequence(A, WClist[i-100:i])
        #https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
        yf = fft(clist) 
        N = len(yf)
        # sample spacing
        T = 1.0 / 800.0
        xf = fftfreq(N, T)[:N//2]
        if n == 0:
            yftot = np.array(yf) # Take abs here or just in the end?
            n+=1
        elif not np.isinf(np.abs(yf[0])):
            yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
            n+=1
    print(n)
    yftot = yftot * (1 / n)
    return xf, yftot, n


def cross_entropy_sequence(A, WClist):
    clist = []
    for i in range(1, len(WClist)):
        clist.append(-np.log(A[class_to_index[WClist[i]]][class_to_index[WClist[i-1]]] + 10**-20))
    return clist
