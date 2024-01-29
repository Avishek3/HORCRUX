import os, time, sys

sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/github/modules")
from modules import rfCommon, datasetGen, S0pt, S0pt_updated
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

################################################

def to_complex(chans):
    '''
    takes in real channels, gets them back to complex
    valued by reversing the processing in fn to_reals()
    '''
    if chans.shape[1]%2 != 0:
        ##print ("ERROR:cannot convert odd #cols to complex")
        return None
    mid = int(chans.shape[1]/2)
    real = chans[:,:mid]
    real = np.array(real)
    imag = chans[:,mid:]
    imag = np.array(imag)
    complex_chan = real+1j*imag
    return complex_chan



def get_array_chans(l1, K, sep, params_list, norm, flag):
    chans = []
    max_norm=[]
    num_chans = len(params_list)
    for i in range(num_chans):
        params = params_list[i]
        x = rfCommon.get_chans_from_params(params, K, sep, l1)
        # ##print(x.shape)

        x = x.transpose()
        # ##print(x.shape)

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
    max_norm = np.array(max_norm)
    max_norm = max_norm.reshape(num_chans,1)
    chans = np.concatenate((chans,max_norm),axis=1)



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
################################################


sep = 0.15  ## antenna separation, no effect when K=1
nfft = 52  ## nfft for channel
K = 1  ## num antennas
cf = 2.412e9  ## center freq
bw = 20e6  ## bandwidth over which channel is observed
l1 = get_lambsWifi(cf, bw, 64)  ## wavelengths, lambda for subcarriers in channel
n_cores = 1  ## number of cores for parallelized data generation, minimum=1
num_paths=1
cf2 = 2.437e9
min_meth = "diff"
window = 20
do_plot=False
l2 = get_lambsWifi(cf2, bw, 64)
nnde_step = .5

snr=[]
error_dist=np.zeros((8,100))

te_params = np.load('github/data/params_whole.npy', allow_pickle=True)

te_X_complex1 = np.load('github/data/te_X_channel_ch1.npy')
te_X_complex1_l2 = np.load('github/data/te_X_channel_l2_ch6.npy')

K = 1
te_X_complex = te_X_complex1[:, nfft * (K - 1):nfft * K]

te_X = datasetGen.to_reals(te_X_complex)

# te_Y=np.load('te_Y_detector_training.npy')

model1 = keras.models.load_model("github/NNmodels/NNDD_25.hdf5")
model2 = keras.models.load_model("github/NNmodels/NNDD_50.hdf5")
model3 = keras.models.load_model("github/NNmodels/NNDD_75.hdf5")
model4 = keras.models.load_model("github/NNmodels/NNDD_100.hdf5")
model5 = keras.models.load_model("github/NNmodels/NNDD_125.hdf5")
model6 = keras.models.load_model("github/NNmodels/NNDD_150.hdf5")
model7 = keras.models.load_model("github/NNmodels/NNDD_175.hdf5")
model8 = keras.models.load_model("github/NNmodels/NNDD_200.hdf5")

pred1 = model1.predict(te_X)
pred2 = model2.predict(te_X)
pred3 = model3.predict(te_X)
pred4 = model4.predict(te_X)
pred5 = model5.predict(te_X)
pred6 = model6.predict(te_X)
pred7 = model7.predict(te_X)
pred8 = model8.predict(te_X)

model_25 = keras.models.load_model("github/NNmodels/NNDE/NNDE_25_ch1.hdf5")
model_50 = keras.models.load_model("github/NNmodels/NNDE/NNDE_50_ch1.hdf5")
model_75 = keras.models.load_model("github/NNmodels/NNDE/NNDE_75_ch1.hdf5")
model_100 = keras.models.load_model("github/NNmodels/NNDE/NNDE_100_ch1.hdf5")
model_125 = keras.models.load_model("github/NNmodels/NNDE/NNDE_125_ch1.hdf5")
model_150 = keras.models.load_model("github/NNmodels/NNDE/NNDE_150_ch1.hdf5")
model_175 = keras.models.load_model("github/NNmodels/NNDE/NNDE_175_ch1.hdf5")
model_200 = keras.models.load_model("github/NNmodels/NNDE/NNDE_200_ch1.hdf5")



########################
for w in range(3,4):
    data_point=100000-2000+w
    d_org, a, b, c = te_params[data_point]
    print(d_org)
    print(a)
    print(b)
    print(c)
    para_check = []
    p = np.array(d_org), np.array(a), np.array(b), np.array(c)
    para_check.append(p)


    ch_l2 = get_array_chans(l2, K, sep, para_check, True, 0)
    no_of_sub = nfft * K
    te_norm = ch_l2[:, no_of_sub]
    ch_l2_complex = ch_l2[:, 0:no_of_sub]
    #ch_l2_complex,e = add_noise_snr_range(ch_l2_complex, 20, 30)

    ch_l1_code = get_array_chans(l1, K, sep, para_check, True, 0)
    no_of_sub = nfft * K
    te_norm = ch_l1_code[:, no_of_sub]
    ch_l1_complex_code = ch_l1_code[:, 0:no_of_sub]
    #ch_l1_complex_code, e = add_noise_snr_range(ch_l1_complex_code, 30,40)



    ###########
    # 25


    pred = pred1[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)


    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 1
    max_d = 25
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d1, g1 = S0pt.pred(model_25, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g1)
    if len(d1)>1:
        d1 = [min(d1)]
    ###########
    # 50

    pred = pred2[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 26
    max_d = 50
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d2, g2 = S0pt.pred(model_50, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g2)
    if len(d2)>1:
        d2 = [min(d2)]
    ###########
    # 75

    pred = pred3[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 51
    max_d = 75
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d3, g3 = S0pt.pred(model_75, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g3)
    if len(d3)>1:
        d3 = [min(d3)]
    ###########
    # 100

    pred = pred4[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 76
    max_d = 100
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d4, g4 = S0pt.pred(model_100, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g4)
    if len(d4)>1:
        d4 = [min(d4)]
    ###########
    # 125

    pred = pred5[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 101
    max_d = 125
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d5, g5 = S0pt.pred(model_125, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g5)
    if len(d5)>1:
        d5 = [min(d5)]
    ###########
    # 150

    pred = pred6[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 126
    max_d = 150
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d6, g6 = S0pt.pred(model_150, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g6)
    if len(d6)>1:
        d6 = [min(d6)]
    ###########
    # 175

    pred = pred7[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 151
    max_d = 175
    distances = np.arange(min_d, max_d, nnde_step)
    ch_l2_guess1, ch_l1_fit1, num_est_paths, d7, g7 = S0pt.pred(model_175, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g7)
    if len(d7)>1:
        d7 = [min(d7)]
    ###########
    # 50

    pred = pred8[data_point, :]
    pred = pred.reshape(1, 1 * nfft * 2)
    pred_complex = to_complex(pred)
    pred_complex_1 = pred_complex.reshape(1, nfft)

    norm_pred_50_max = np.max(np.abs(pred_complex_1[0, :]))
    norm_pred_50 = pred_complex_1[0, :] / norm_pred_50_max

    ch_l1_50 = norm_pred_50
    ch_l1_50 = ch_l1_50.reshape(nfft, 1)

    min_d = 176
    max_d = 200
    distances = np.arange(min_d, max_d, nnde_step)

    ch_l2_guess1, ch_l1_fit1, num_est_paths, d8, g8 = S0pt.pred(model_200, K, ch_l1_50, l1, l2, sep, min_d, max_d,
                                                                min_meth, window, distances, do_plot)
    print(g8)
    if len(d8)>1:
        d8 = [min(d8)]
## whole
    min_d = 1
    max_d = 200

    d_est = np.concatenate((d1, d2, d3,d4,d5,d6,d7,d8), axis=0)
    g_est = np.concatenate((g1,g2,g3,g4,g5,g6,g7,g8),axis=0)

    np.save('g_est',g_est)


    distances = np.arange(1, 200, .5)

    pred_complex_final = te_X_complex[data_point, :]

    norm_pred_final_max = np.max(np.abs(pred_complex_final))
    norm_pred_final = pred_complex_final / norm_pred_final_max
    ch_l1_final = norm_pred_final

    ch_l1_final = pred_complex_final.reshape(nfft,1) #ch_l1_final.reshape(nfft, 1)

    ch_l2_guess_final, ch_l1_fit_final, num_est_paths, d_final = S0pt_updated.pred(ch_l1_complex_code.reshape(nfft,1), l1, l2, sep, min_d,
                                                                                    max_d, min_meth, window, d_est)
    #ch_l1_complex_code.reshape(nfft,1)#
    ch_l2_guess_final = ch_l2_guess_final / (np.max(np.abs(ch_l2_guess_final)))
    ch_l1_fit_final = ch_l1_fit_final / np.max(np.abs(ch_l1_fit_final))

    print('D_orginal.....', d_org)
    print('D_opt', d_final)
    print(d_est)

    ch_l2_org = te_X_complex1_l2[data_point,:]


    error = np.abs(ch_l2_guess_final.transpose() - ch_l2_complex)
    # #print(ch2_pred_whole.shape)

    mean_error = np.mean(pow(error, 2))

    # #print(mean_error)
    csi_mean = np.mean(pow(np.abs(ch_l2_complex), 2))
    # #print(csi_mean)

    snr.append(-10 * np.log10(mean_error / csi_mean))

    #
    fig = plt.figure()
    plt.plot(np.real(ch_l2_guess_final),'k--')
    plt.plot(np.real(ch_l2_complex[0,:]),'r')
    plt.legend(['pred','org'])
    plt.savefig('github/Fig/comp.png')

    fig = plt.figure()
    plt.plot(np.angle(ch_l2_guess_final), 'k--')
    plt.plot(np.angle(ch_l2_complex[0, :]), 'r')
    plt.legend(['pred', 'org'])
    plt.savefig('github/Fig/comp2.png')
    fig = plt.figure()
    plt.plot(np.real(ch_l1_fit_final))
    # plt.plot(np.abs(norm_pred_final))
    plt.plot(np.real(ch_l1_complex_code[0,:]),'k--')
    plt.legend(['pred','data'])
    plt.savefig('github/Fig/comp1.png')

    # print((np.round(np.real(ch_l2_guess_final),2)) + 1j*np.round(np.imag(ch_l2_guess_final),2) )
    # print((np.round(np.real(ch_l2_complex[0,:]),2)) + 1j*np.round(np.imag(ch_l2_complex[0,:]),2) )
    # print((np.round(np.real(ch_l1_fit_final),2)) + 1j*np.round(np.imag(ch_l1_fit_final),2))
    # print((np.round(np.real(ch_l1_complex_code[0, :]), 2)) + 1j*np.round(np.imag(ch_l1_complex_code[0, :]), 2))




b=np.array(snr)
x1 = np.sort(b)
y1 = np.arange(len(x1))/float(len(x1))
fig=plt.figure()
plt.plot(x1, y1,'r')

plt.xlabel('SNR dB')
plt.ylabel('cdf')
plt.savefig('github/Fig/snr2-8secs_NNDD_20mhz_wifi')