import os, time, sys

sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/github/modules")

import numpy as np
from modules import mlDE, rfCommon, datasetGen
import keras

import pylab as pl
import matplotlib.pyplot as plt
from scipy import sparse

#########################################################
def add_noise_snr_range(chans, min_snr, max_snr):
    n = chans.shape[0]
    r,c = chans.shape
    snrs = np.random.randint(min_snr, max_snr+1, n)
    scale = np.power(10, -1*snrs/20.0)
    scale = scale.reshape([-1,1])
    noise = np.random.randn(r,c)+1j*np.random.randn(r,c)
    noise = noise/np.abs(noise)
    noise = scale*noise
    chans = chans+noise/4
    return chans

def plot_1D_decompose_pred(Y_act, Y_pred, name,params):
    pl.figure(figsize=(15, 10))
    for i in range(10):
        d,a,b,c = params[i]
        print(d)
        pl.subplot(5, 2, i + 1)
        x = Y_act[i][0]
        print(x)
        x = x.ravel().transpose()
        pl.plot(x)
        pl.plot(Y_pred[i])
        if i == 0:
            leg = ["Actual component positions", "Estimated"]
            pl.legend(leg)
    if name != "":
        pl.savefig(name)
        pl.close()
    else:
        pl.show()

def model_NNDE(model_name,tr_X,tr_Y,te_X,te_Y):
    h_units = 200
    num_h_layers = 5
    sigma = 1

    cf_name = str(round(cf / 1e9, 2))  ## cf in string format for name
    bw_name = str(round(bw / 1e6, 2))  ## bw in string format for name
    name = cf_name + "_d" + str(max_d) + "_bw" + bw_name + "MHz" + "_1d_sigma_" + str(sigma)
    name = model_name
    '''
    begin training NNDE
    '''
    ### NN parameters
    activation = "elu"
    act_in = "elu"
    act_out = "elu"
    optimizer = "adam"
    lr = 0.001
    batch_size = 256
    num_epochs = 150
    loss_name = "mean_absolute_error"
    num_chans = tr_X.shape[0]
    steps_per_epoch = num_chans / batch_size
    in_dim = tr_X.shape[1]
    out_dim = tr_Y.shape[1]

    sigma = sigma * max(1, te_Y.shape[1] / 50)
    d = str(te_Y.shape[1])
    conv_filter = sigma

    fname = "github/NNmodels/NNDE/" + name + ".hdf5"
    callback_list = mlDE.get_callbacks(0.001, fname, 150, 0.01)
    print("setting:", name)

    model = mlDE.get_model(in_dim, out_dim, h_units, num_h_layers, activation, optimizer, loss_name, act_in, act_out,
                              learning_rate=lr)

    history = model.fit_generator(
        datasetGen.nn_batch_generator_convolved(tr_X, tr_Y, batch_size, conv_filter),
        steps_per_epoch,
        epochs=num_epochs,
        verbose=2,
        validation_data=datasetGen.nn_batch_generator_convolved(te_X, te_Y, batch_size, conv_filter),
        validation_steps=1,
        callbacks=callback_list
    )
    return model, name

#########################################################
if __name__ == '__main__':
    ### RF and antenna parameters
    sep = 0.06  ## antenna separation, no effect when K=1
    nfft = 52  ## nfft for channel
    K = 1  ## num antennas
    cf = 2.412e9  ## center freq
    bw = 20e6  ## bandwidth over which channel is observed
    l1 = rfCommon.get_lambs(cf, bw, nfft)  ## wavelengths, lambda for subcarriers in channel
    min_amp = 0.05


    '''
    generate data for training and testing
    '''
    zones =[25,50,75,100,125,150,175,200]
    min_d_list = [ 1,26,51,76,101,126,151,176] 
    max_d_list = [25,50,75,100,125,150,175,200]

    for i in range(len(zones)):
        z= zones[i]
        min_d = min_d_list[i] ## min travel distance for any channel component, meters
        max_d = max_d_list[i] ## max travel distance for any channel component, meters
        nnde_step = .5   ## for output of NNDE, the quantization step for distances, meters
        distances = np.arange(min_d, max_d, nnde_step)  ## list of distances used for output of NNDE

        whole_param_list = np.load('github/data/params_list_'+str(z)+'.npy', allow_pickle=True)
        num_chans = len(whole_param_list) - 2000

        params_list = whole_param_list[0:num_chans, :]
        te_params = whole_param_list[num_chans:, :]

        no_of_sub = nfft * K

        ch_50 = np.load('github/data/te_X_channel_'+str(z)+'_ch1.npy')
        ch_50_complex = ch_50[0:num_chans, 0:no_of_sub]


        tr_X_complex = ch_50_complex
        te_X_complex = ch_50[num_chans:, 0:no_of_sub]


        # tr_X_complex = add_noise_snr_range(tr_X_complex, 20, 30)
        # te_X_complex = add_noise_snr_range(te_X_complex, 20, 30)

        tr_X = datasetGen.to_reals(tr_X_complex)
        te_X = datasetGen.to_reals(te_X_complex)

        tr_Y = datasetGen.get_sparse_target(params_list, distances, min_amp)
        te_Y = datasetGen.get_sparse_target(te_params, distances, min_amp)

        data = tr_X, tr_Y, te_X, te_Y
        
        tr_X, tr_Y, te_X, te_Y = data
        print("I/O shapes:", tr_X.shape, tr_Y.shape, te_X.shape, te_Y.shape)
        
        model,name = model_NNDE('NNDE_'+str(z)+'_ch1',tr_X,tr_Y,te_X,te_Y)

        y_act = te_Y.todense()
        print(y_act.shape)
        y_pred = model.predict(te_X)
        plot_1D_decompose_pred(y_act, y_pred, "github/Fig/" + name + ".png",te_params)

    

    


