import numpy as np
import os, time, sys
sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/modules")


from modules import rfCommon, datasetGen
sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/modules/simopt")
from modules.simopt import S0pt
import pylab as pl

def subbed_pred(model, K, ch_l1, l1, l2, sep, min_d, max_d, min_meth, window, distances, version):
    '''
    model = keras NN model object
    K = number of antennae
    ch_l1 = channel matrix in fequency band 1, nfft x K
    l1 = lambda1, wavelengths used for OFDM in band 1
    l2 = wavelengths used for OFDM in band 2
    sep = separation between antennas
    min_d = min distance of reflectors
    nax_d = max distance of reflectors
    min_meth = method used for miminization of objective function
    window = given a distance estimate d, search a better fit within d+/- window
    distances = list of distances, corresponding to each output neuron of the nnde
    version = variation of optimization.
    '''
    rots = np.arange(-1*sep, sep, sep/10)
    tots = np.zeros(len(distances))
    dats = []
    th = 0.6
    if K == 1:
        dat = datasetGenen.to_reals(ch_l1[:,0]).transpose()
        g = model.predict(dat)[0]
        g = g/np.max(g)
        g = np.where(g<th,0,g)
        ds = rfCommon.get_peak_distances(g, distances)
    else:
        for n in range(len(rots)):
            k = np.random.randint(1,K)
            rot = np.exp(-2j*np.pi*k*rots[n]/l1)
            ch_rot = ch_l1[:,k]*rot
            s = ch_l1[:,0]-ch_rot
            dats.append(s)
        dats = np.array(dats)
        dats = datasetGen.to_reals(dats)
        g = model.predict(dats)
        tots = g.sum(0)
        # pl.plot(tots)
        # pl.show()
        tots = tots/np.max(tots)
        tots = np.where(tots<th,0,tots)
        ds = rfCommon.get_peak_distances(tots, distances)
    if len(ds)==0:
        ds = [max_d/2]
    if len(ds)>4:
        ds = ds[:4]
    #pl.plot(g.transpose())
    #pl.show()

    if version == "simple":
        ch_l1_guess, ch_l2_guess = S0pt.fit_simple(ch_l1, l1, l2, sep, min_d, max_d, min_meth, ds, window)
    elif version == "avg":
        ch_l1_guess, ch_l2_guess = S0pt.fit_simple_averaged(ch_l1, l1, l2, sep, min_d, max_d, min_meth, ds, window)
    elif version == "K0":
        ch_l1_guess, ch_l2_guess = S0pt.fit_simple_K0(ch_l1, l1, l2, sep, min_d, max_d, min_meth, ds, window)
    elif version == "bestK":
        ch_l1_guess, ch_l2_guess = S0pt.fit_simple_bestK(ch_l1, l1, l2, sep, min_d, max_d, min_meth, ds, window)
    return ch_l2_guess, len(ds)
