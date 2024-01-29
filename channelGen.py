import os, time, sys

sys.path.insert(0, "/Users/avibaner/Downloads/Optml_avishek/github/modules")
from modules import rfCommon, datasetGen
import numpy as np
import multiprocessing
from multiprocessing import Pool
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

##########################################################################
# Define the functions required to generate channels

def get_array_chans_multi_proc(lambs, K, sep, params_list, n_processes, norm, flag):
    '''
    helper function to speed up generation of channels based on parameters.
    multiple processors are used and their results are combined.
    '''
    p = Pool(processes=n_processes)
    num_chans = len(params_list)
    num_chans_per_proc = num_chans // n_processes
    results = []
    for i in range(n_processes):
        start = i * num_chans_per_proc
        end = min(start + num_chans_per_proc, num_chans)
        sub_params_list = params_list[start:end]
        results.append(p.apply_async(get_array_chans, args=(lambs, K, sep, sub_params_list, norm, flag)))
    p.close()
    p.join()

    output = [p.get() for p in results]
    X = None
    for i in range(0, len(output)):
        x = output[i]
        if i == 0:
            X = x
        else:
            X = np.vstack([X, x])
    del results, output
    return X


def get_array_chans(l1, K, sep, params_list, norm, flag):
    chans = []
    max_norm=[]
    num_chans = len(params_list)
    for i in range(num_chans):
        params = params_list[i]
        x = rfCommon.get_chans_from_params(params, K, sep, l1)
        # print(x.shape)

        x = x.transpose()
        # print(x.shape)

        x_flat = x.ravel()
        x_max = np.max(np.abs(x_flat))

        if flag == 1:
            if norm:
                x = x / x_max
            chans.append(x)
        else:
            if norm:
                x_flat = x_flat / x_max
            chans.append(x_flat)
            max_norm.append(x_max)
    chans = np.array(chans)
    # max_norm = np.array(max_norm)
    # max_norm = max_norm.reshape(num_chans,1)
    # chans = np.concatenate((chans,max_norm),axis=1)



    return chans

def add_noise_snr_range(chans, min_snr, max_snr):
    n = chans.shape[0]
    r,c = chans.shape
    snrs = np.random.randint(min_snr, max_snr+1, n)
    scale = np.power(10, -1*snrs/20.0)
    scale = scale.reshape([-1,1])
    noise = np.random.randn(r,c)+1j*np.random.randn(r,c)
    noise = noise/np.abs(noise)
    noise = scale*noise
    chans = chans+noise
    return chans, noise

def get_lambsWifi(cf, bw, nfft):
    c = 3e8
    f = np.fft.fftfreq(nfft)
    f = np.fft.fftshift(f)
    f = f * bw
    cf = f + cf
    cf_wifi = np.concatenate((cf[6:32], cf[33:59]))
    l1 = []
    for f in cf_wifi:
        l1.append(c / f)
    return np.array(l1)

################################################

if __name__ == '__main__':

    ### RF and antenna parameters
    sep = 0.06  ## antenna separation, no effect when K=1
    nfft = 52  ## nfft for channel
    K = 1  ## num antennas
    cf = 2.412e9  ## center freq
    bw = 20e6  ## bandwidth over which channel is observed
    l1 = get_lambsWifi(cf, bw, 64)  ## wavelengths, lambda for subcarriers in channel
    d_sep = None  ## for generating channels, what is the min separation between any two components
    n_cores = 4
    num_chans = 100000
    cf2 = 2.437e9
    l2 = get_lambsWifi(cf2, bw, 64)



    te_params = np.load('github/data/params_whole.npy',allow_pickle=True)
    te_params_50 = np.load('github/data/params_list_50.npy',allow_pickle=True)
    te_params_100 = np.load('github/data/params_list_100.npy',allow_pickle=True)
    te_params_150 = np.load('github/data/params_list_150.npy',allow_pickle=True)
    te_params_200 = np.load('github/data/params_list_200.npy',allow_pickle=True)
    te_params = np.load('github/data/params_whole.npy',allow_pickle=True)
    te_params_25 = np.load('github/data/params_list_25.npy',allow_pickle=True)
    te_params_75 = np.load('github/data/params_list_75.npy',allow_pickle=True)
    te_params_125 = np.load('github/data/params_list_125.npy',allow_pickle=True)
    te_params_175 = np.load('github/data/params_list_175.npy',allow_pickle=True)


    ch_25 = get_array_chans_multi_proc(l1, K, sep, te_params_25, n_cores, True, 0)
    print('channel done')
    ch_25,d = add_noise_snr_range(ch_25, 20, 30)

    ch_25_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_25, n_cores, True, 0)
    ch_25_l2 = ch_25_l2 + d

    ch_50 = get_array_chans_multi_proc(l1, K, sep, te_params_50, n_cores, True, 0)
    print('channel done')
    ch_50,d = add_noise_snr_range(ch_50, 20, 30)
    ch_50_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_50, n_cores, True, 0)
    ch_50_l2 = ch_50_l2 + d



    ch_75 = get_array_chans_multi_proc(l1, K, sep, te_params_75, n_cores, True, 0)
    print('channel done')
    ch_75,d = add_noise_snr_range(ch_75, 20, 30)
    ch_75_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_75, n_cores, True, 0)
    ch_75_l2 = ch_75_l2 + d



    ch_100 = get_array_chans_multi_proc(l1, K, sep, te_params_100, n_cores, True, 0)
    print('channel done')

    ch_125 = get_array_chans_multi_proc(l1, K, sep, te_params_125, n_cores, True, 0)
    print('channel done')
    ch_125,d = add_noise_snr_range(ch_125, 20, 30)
    ch_125_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_125, n_cores, True, 0)
    ch_125_l2 = ch_125_l2 + d


    ch_150 = get_array_chans_multi_proc(l1, K, sep, te_params_150, n_cores, True, 0)
    print('channel done')

    ch_175 = get_array_chans_multi_proc(l1, K, sep, te_params_175, n_cores, True, 0)
    print('channel done')
    ch_175,d = add_noise_snr_range(ch_175, 20, 30)
    ch_175_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_175, n_cores, True, 0)
    ch_175_l2 = ch_175_l2 + d


    ch_200 = get_array_chans_multi_proc(l1, K, sep, te_params_200, n_cores, True, 0)
    print('channel done')


    ch_100,d = add_noise_snr_range(ch_100, 20, 30)
    ch_100_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_100, n_cores, True, 0)
    ch_100_l2 = ch_100_l2 + d

    ch_150,d = add_noise_snr_range(ch_150, 20, 30)
    ch_150_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_150, n_cores, True, 0)
    ch_150_l2 = ch_150_l2 + d


    ch_200,d = add_noise_snr_range(ch_200, 20, 30)
    ch_200_l2 = get_array_chans_multi_proc(l2, K, sep, te_params_200, n_cores, True, 0)
    ch_200_l2 = ch_200_l2 + d

    ch_whole= ch_50+ch_100+ch_150+ch_200+ch_25+ch_75+ch_125+ch_175
    ch_whole_l2=ch_50_l2+ch_100_l2+ch_150_l2+ch_200_l2+ch_25_l2+ch_75_l2+ch_125_l2+ch_175_l2
    ch_whole_max=np.amax(np.abs(ch_whole), axis=1).reshape(num_chans,1)
    ch_whole_n=ch_whole/ch_whole_max

    ch_50_n = ch_50
    # ch_50_max=np.amax(np.abs(ch_50), axis=1).reshape(num_chans,1)
    # ch_50=ch_50/ch_50_max

    # ch_100_max=np.amax(np.abs(ch_100), axis=1).reshape(num_chans,1)
    # ch_100=ch_100/ch_100_max
    #
    # ch_150_max=np.amax(np.abs(ch_150), axis=1).reshape(num_chans,1)
    # ch_150=ch_150/ch_150_max
    #
    # ch_200_max=np.amax(np.abs(ch_200), axis=1).reshape(num_chans,1)
    # ch_200=ch_200/ch_200_max


    np.save('github/data/te_X_channel_ch1',ch_whole_n)
    np.save('github/data/te_X_channel_50_ch1',ch_50)
    np.save('github/data/te_X_channel_100_ch1',ch_100)
    np.save('github/data/te_X_channel_150_ch1',ch_150)
    np.save('github/data/te_X_channel_200_ch1',ch_200)
    np.save('github/data/te_X_channel_25_ch1',ch_25)
    np.save('github/data/te_X_channel_75_ch1',ch_75)
    np.save('github/data/te_X_channel_125_ch1',ch_125)
    np.save('github/data/te_X_channel_175_ch1',ch_175)

    np.save('github/data/te_X_channel_l2_ch6',ch_whole_l2)
    np.save('github/data/te_X_channel_50_l2_ch6',ch_50_l2)
    np.save('github/data/te_X_channel_100_l2_ch6',ch_100_l2)
    np.save('github/data/te_X_channel_150_l2_ch6',ch_150_l2)
    np.save('github/data/te_X_channel_200_l2_ch6',ch_200_l2)
    np.save('github/data/te_X_channel_25_l2_ch6',ch_25_l2)
    np.save('github/data/te_X_channel_75_l2_ch6',ch_75_l2)
    np.save('github/data/te_X_channel_125_l2_ch6',ch_125_l2)
    np.save('github/data/te_X_channel_175_l2_ch6',ch_175_l2)

    # data_point = 10
    # ch_test1 = ch_25[data_point,:]
    # ch_test2 = ch_50[data_point,:]
    # ch_test3 = ch_75[data_point,:]
    # ch_test4 = ch_125[data_point,:]
    # ch_test5 = ch_150[data_point,:]
    # ch_test6 = ch_200[data_point,:]

    # ch_test_n = ch_25_n[data_point,:]
    #
    # fig = plt.figure()
    # plt.plot(np.abs(ch_whole_n),'b')
    # # plt.plot(np.abs(ch_test2),'r')
    # # plt.plot(np.abs(ch_test3),'b--')
    # # plt.plot(np.abs(ch_test4),'r--')
    # # plt.plot(np.abs(ch_test5),'g')
    # plt.plot(np.abs(ch_whole),'k--')
    # # plt.plot(np.abs(ch_test_n),'r')
    # # plt.legend(['25','50','75','125','150','200'])
    # plt.savefig('Fig/norm_check.png')

