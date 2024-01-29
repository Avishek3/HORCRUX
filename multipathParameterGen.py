import os, time, sys

sys.path.insert(0, "Users/avibaner/Downloads/Optml_avishek/github/modules")
from modules import rfCommon, datasetGen
import numpy as np

print("Start generating........")

##########################################################################
# Define the functions required to generate channel parameters

def add_params(p1,p2,p3,p4,p5,p6,p7,p8):
    num_chans = len(p1)
    params_final=[]
    for i in range(num_chans):
        p_1=p1[i]
        p_2=p2[i]
        p_3=p3[i]
        p_4=p4[i]
        p_5=p5[i]
        p_6=p6[i]
        p_7=p7[i]
        p_8=p8[i]
        d_ns, a_ns, phi_ns, psi_ns = append_params(p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8)
        p = np.array(d_ns), np.array(a_ns), np.array(phi_ns), np.array(psi_ns)
        params_final.append(p)
    return params_final


def append_params(p1,p2,p3,p4,p5,p6,p7,p8):
    d=[]
    a=[]
    b=[]
    c=[]
    d1,a1,b1,c1 = p1
    d2, a2, b2, c2 = p2
    d3, a3, b3, c3 = p3
    d4, a4, b4, c4 = p4
    d5,a5,b5,c5 = p5
    d6, a6, b6, c6 = p6
    d7, a7, b7, c7 = p7
    d8, a8, b8, c8 = p8
    d=np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8),axis=0)
    a = np.concatenate((a1, a2,a3,a4,a5,a6,a7,a8), axis=0)
    b = np.concatenate((b1, b2,b3,b4,b5,b6,b7,b8), axis=0)
    c = np.concatenate((c1, c2,c3,c4,c5,c6,c7,c8), axis=0)
    return d,a,b,c

################################################


if __name__ == '__main__':
    ### RF and antenna parameters
    sep = 0.15  ## antenna separation, no effect when K=1
    nfft = 64  ## nfft for channel
    K = 1  ## num antennas
    cf = 2.4e9  ## center freq
    bw = 20e6  ## bandwidth over which channel is observed
    l1 = rfCommon.get_lambs(cf, bw, nfft)  ## wavelengths, lambda for subcarriers in channel
    #### channel parameters

    d_sep = None  ## for generating channels, what is the min separation between any two components
    min_n_paths = 1  ## min num paths
    max_n_paths = 1  ## max num paths
    min_amp = 0.05  ## min amplitude of any multipath component
    num_chans = 100000  ## num of channels to generate
    n_cores = 4  ## number of cores for parallelized data generation, minimum=1

    # Get parameter list for dist 1 to 25
    min_d = 1  ## min travel distance for any channel component, meters
    max_d = 25  ## max travel distance for any channel component, meters
    nnde_step = 1  ## for output of NNDE, the quantization step for distances, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_25 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    print('Finishing step1')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_25', params_list_25)

    # Get parameter list for dist 26 to 25
    min_d = 26  ## min travel distance for any channel component, meters
    max_d = 50  ## max travel distance for any channel component, meters
    nnde_step = 1  ## for output of NNDE, the quantization step for distances, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_50 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    print('Finishing step2')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_50', params_list_50)

    # Get parameter list for dist 51 to 75

    min_d = 51  ## min travel distance for any channel component, meters
    max_d = 75  ## max travel distance for any channel component, meters
    nnde_step = 1  ## for output of NNDE, the quantization step for distances, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_75 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    print('Finishing step3')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_75', params_list_75)

     # Get parameter list for dist 76 to 100
    min_d = 76  ## min travel distance for any channel component, meters
    max_d = 100  ## max travel distance for any channel component, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_100 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths,
                                                       d_sep)
    print('Finishing step4')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_100', params_list_100)

     # Get parameter list for dist 101 to 125
    min_d = 101  ## min travel distance for any channel component, meters
    max_d = 125  ## max travel distance for any channel component, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_125 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths,
                                                       d_sep)
    print('Finishing step5')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_125', params_list_125)



    # Get parameter list for dist 126 to 150
    min_d = 126  ## min travel distance for any channel component, meters
    max_d = 150  ## max travel distance for any channel component, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_150 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths,
                                                       d_sep)
    print('Finishing step6')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_150', params_list_150)

    
    # Get parameter list for dist 151 to 175
    min_d = 151  ## min travel distance for any channel component, meters
    max_d = 175  ## max travel distance for any channel component, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_175 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths,
                                                       d_sep)
    print('Finishing step7')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_175', params_list_175)

    # Get parameter list for dist 176 to 200
    min_d = 176  ## min travel distance for any channel component, meters
    max_d = 200  ## max travel distance for any channel component, meters
    distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE
    params_list_200 = datasetGen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths,d_sep)
    print('Finishing step8')
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_list_200', params_list_200)

    params_whole = add_params(params_list_25,params_list_50,params_list_75, params_list_100,params_list_125, params_list_150,params_list_175, params_list_200)
    np.save('/Users/avibaner/Downloads/Optml_avishek/github/data/params_whole', params_whole)
