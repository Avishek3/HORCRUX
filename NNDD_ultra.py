import os, time, sys

sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/github/modules")
import numpy as np
from modules import datasetGen, mlDD
import matplotlib.pyplot as plt
########################################

def to_complex(chans):
    '''
    takes in real channels, gets them back to complex
    valued by reversing the processing in fn to_reals()
    '''
    if chans.shape[1]%2 != 0:
        print ("ERROR:cannot convert odd #cols to complex")
        return None
    mid = int(chans.shape[1]/2)
    real = chans[:,:mid]
    real = np.array(real)
    imag = chans[:,mid:]
    imag = np.array(imag)
    complex_chan = real+1j*imag
    return complex_chan

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

def train_NNDD(model_name, tr1_X, tr1_Y, te1_X, te1_Y):
    h_units = 512
    num_h_layers = 2
    sigma = 1

    cf_name = str(round(cf / 1e9, 2))  ## cf in string format for name
    bw_name = str(round(bw / 1e6, 2))  ## bw in string format for name
    name = model_name

    '''
    begin training NNDE
    '''
    ## NN parameters
    activation = "elu"
    act_in = "elu"
    act_out = "elu"
    optimizer = "adam"
    lr = 0.001
    batch_size = 256
    num_epochs = 250
    loss_fn = loss_name = "mean_absolute_error"
    num_chans = tr1_X.shape[0]
    steps_per_epoch = num_chans / batch_size
    in_dim = tr1_X.shape[1]
    out_dim = tr1_Y.shape[1]

    sigma = sigma * max(1, te1_Y.shape[1] / 200)
    d = str(te_Y.shape[1])
    conv_filter = sigma

    fname = "github/NNmodels/" + name + ".hdf5"
    callback_list = mlDD.get_callbacks(0.001, fname, 250, 0.01)
    print("setting:", name)


    model_detector = mlDD.get_model(in_dim, out_dim, h_units, num_h_layers, activation, optimizer, loss_name, act_in, act_out,
                              learning_rate=lr)


    history= model_detector.fit(
        tr1_X, tr1_Y,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(te1_X, te1_Y),
        callbacks = callback_list
    )



########################################

num_chans=100000
nfft = 52
K = 1
cf = 2.412e9  ## center freq
bw = 20e6  ## bandwidth over which channel is observed


te_X_whole= np.load('github/data/te_X_channel_ch1.npy')
no_of_sub=nfft*K
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_25_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')
#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_25',tr1_X, tr1_Y, te1_X, te1_Y)

######################################################################################
#50
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_50_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')

#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)
partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]
train_NNDD('NNDD_50',tr1_X, tr1_Y, te1_X, te1_Y)

#######################################################################################
#75
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_75_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')

#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)
partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_75',tr1_X, tr1_Y, te1_X, te1_Y)


######################################################################################
#100
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_100_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')


#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_100',tr1_X, tr1_Y, te1_X, te1_Y)


######################################################################################
#125
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_125_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')

#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_125',tr1_X, tr1_Y, te1_X, te1_Y)


######################################################################################
#150
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_150_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')

#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_150',tr1_X, tr1_Y, te1_X, te1_Y)


######################################################################################
#175
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_175_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]

te_Y_complex = ch_50_complex

print('Done loading')


#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_175',tr1_X, tr1_Y, te1_X, te1_Y)


######################################################################################
#200
te_X_complex=te_X_whole[0:num_chans,0:no_of_sub]
#te_X_complex, te_noise = add_noise_snr_range(te_X_complex, 20, 30)

ch_50=np.load('github/data/te_X_channel_200_ch1.npy')
ch_50_complex=ch_50[0:num_chans,0:no_of_sub]
te_Y_complex = ch_50_complex

print('Done loading')


#######################################

te_X = datasetGen.to_reals(te_X_complex)
te_Y= datasetGen.to_reals(te_Y_complex)

partition=num_chans-2000

tr1_X=te_X[0:partition,:]
tr1_Y=te_Y[0:partition,:]

te1_X=te_X[partition:,:]
te1_Y=te_Y[partition:,:]

train_NNDD('NNDD_200',tr1_X, tr1_Y, te1_X, te1_Y)

