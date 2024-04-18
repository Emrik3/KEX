from scipy.fft import fft, fftfreq
from choose_word_classes import class_to_index
import numpy as np
from metrics import tensorfrobnorm
import scipy
import random
import torch
import copy
#from torchmetrics.image import SpectralAngleMapper

def fourier_test(A, WClist):
    # Need to test these so that I know what is going on, changing the n_values variables is weird...
    n = 0
    yftot = []
    nwords = 20
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
            elif not np.isinf(yf[0].any()):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
    print(n)
    yftot = yftot * (1 / n)
    return xf, yftot, n

def fourier_test_for_bible(A, WClist, stop_at_val=False):
    # This is what need to be done in all these files and with the WC list inputed into them!
    """More words are needed for each run, it works if we run full abstracts and avrage of those but also need to avrage
    over the newspaper, this is quite hard to do. maybe leave til the end, if we need to start writing about this we start
    by saying what we have done and then if we don't have tome to fix we say that what we did, did not work and leave it 
    at further work"""
    n = 0
    yftot = []
    nwords = 20
    WC_long = []
    k = 0
    for i in range(len(WClist)):
        WC_long.append(WClist[i])
        k += 1
        if WClist[i] == '.':
            k = 0
            WC_long = []
        if k == nwords:
            #print(WC_long)
            clist = cross_entropy_sequence(A, WC_long)
            #https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
            yf = fft(clist) 
            N = len(yf)
            # sample spacing
            T = 1.0 / 800.0
            xf = fftfreq(N, T)[:N//2]
            if n == 0:
                yftot = np.array(yf) # Take abs here or just in the end?
                n+=1
            elif not np.isinf(yf[0].all()):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
            if stop_at_val and n == 4008:
                yftot = yftot * (1 / n)
                print(n)
                return xf, yftot, n
    
    yftot = yftot * (1 / n)
    return xf, yftot, n

def fourier_test_shuffle_bible(A, WClist, stop_at_val=False):
    # This is what need to be done in all these files and with the WC list inputed into them!
    """More words are needed for each run, it works if we run full abstracts and avrage of those but also need to avrage
    over the newspaper, this is quite hard to do. maybe leave til the end, if we need to start writing about this we start
    by saying what we have done and then if we don't have tome to fix we say that what we did, did not work and leave it 
    at further work"""
    n = 0
    yftot = []
    nwords = 20
    WC_long = []
    k = 0
    ll = copy.deepcopy(WClist)
    random.shuffle(ll)
    for i in range(len(ll)):
        WC_long.append(ll[i])
        k += 1
        if ll[i] == '.':
            k = 0
            WC_long = []
        if k == nwords:
            #print(WC_long)
            
            clist = cross_entropy_sequence(A, WC_long)
            #https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
            yf = fft(clist) 
            N = len(yf)
            # sample spacing
            T = 1.0 / 800.0
            xf = fftfreq(N, T)[:N//2]
            if n == 0:
                yftot = np.array(yf) # Take abs here or just in the end?
                n+=1
            elif not np.isinf(yf[0].all()):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
            if stop_at_val and n == 4008:
                yftot = yftot * (1 / n)
                return xf, yftot, n
    yftot = yftot * (1 / n)
    
    return xf, yftot, n

def fourier_test_for_1990(A, WClist):
    # This is what need to be done in all these files and with the WC list inputed into them!
    """More words are needed for each run, it works if we run full abstracts and avrage of those but also need to avrage
    over the newspaper, this is quite hard to do. maybe leave til the end, if we need to start writing about this we start
    by saying what we have done and then if we don't have tome to fix we say that what we did, did not work and leave it 
    at further work"""
    n = 0
    yftot = []
    nwords = 20
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
            elif not np.isinf(yf[0].all()):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
    print(n)
    yftot = yftot * (1 / n)
    return xf, yftot, n

def fouriertest_shuffla(A, WClist):
    n = 0
    yftot = []
    nwords = 20
    ll = WClist
    random.shuffle(ll)
    for WC in ll:
        if len(WC) >= nwords:
            clist = cross_entropy_sequence(A, ll)
            #https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
            yf = fft(clist) 
            N = len(yf)
            # sample spacing
            T = 1.0 / 800.0
            xf = fftfreq(N, T)[:N//2]
            if n == 0:
                yftot = np.array(yf) # Take abs here or just in the end?
                n+=1
            elif not np.isinf(yf[0].any()):
                yftot = np.add(yftot, np.array(yf)) # Take abs here or just in the end?
                n+=1
    yftot = yftot * (1 / n)
    print(n)
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

def cross_entropy_sequence_mult(A, WClist):
    # Not corret but might work as an assumtion to what is correct... 
    clist = []
    llist = []
    for i in range(1, len(WClist)):
        if i==1:
            llist.append(A[class_to_index[WClist[i]]][class_to_index[WClist[i-1]]] + 10**-100)
        else:
            llist.append(A[class_to_index[WClist[i]]][class_to_index[WClist[i-1]]] * llist[i-2] + 10**-100)
        clist.append(-np.log(llist[i-1]))
    #print(llist)
    return clist

# Make test for shuffled thing, and use cross_entropy_sequence_mult, this is kind of good but if this would be done with multiple orders it might be better.

def cross_entropy_sequence_mult2(A, WClist):
    # Order two and multiplying
    clist = []
    llist = []
    for i in range(2, len(WClist)):
        if i==2:
            llist.append(A[class_to_index[WClist[i]]][class_to_index[WClist[i-1]]][class_to_index[WClist[i-2]]] + 10**-30)
        else:
            llist.append(A[class_to_index[WClist[i]]][class_to_index[WClist[i-1]]][class_to_index[WClist[i-2]]] * llist[i-3]  + 10**-30)
        clist.append(-np.log(llist[i-2]))
    return clist



def pearson_corr_coeff(X, Y):
    #return np.cov(X, Y) / (np.std(X) * np.std(Y))
    #return np.corrcoef(X, Y) # This is wrong function?
    return (np.mean(np.multiply(X, Y)) - np.mean(X)*np.mean(Y)) / (np.std(X)*np.std(Y))


def spearman_corr_coeff(X, Y):
    # Gives same bad result, because too few input things... 
    return scipy.stats.spearmanr(X, Y)

def spec_ang_map(X, Y):
    pass
    # Not working
    #sam = SpectralAngleMapper()
    #return sam(X, Y)

def dist_corr(X, Y):
    return scipy.stats.somersd(X, Y)

# What works: scipy.stats.kendalltau(X, Y), scipy.stats.somersd(X, Y), see: https://docs.scipy.org/doc/scipy/reference/stats.html

# I think that the 1990 file is not going to work, but shuffled is good. 


def AUC(X, Y):
    # Area under curve.
    pass
    
