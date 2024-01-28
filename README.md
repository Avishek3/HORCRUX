# HORCRUX
Code base for HORCRUX
## Edit from terminal

Parameters required to generate wireless channels. Please change the values accordingly to generate channels
```
### RF and antenna parameters
    sep = 0.06  ## antenna separation, no effect when K=1
    nfft = 52  ## nfft for channel
    K = 1  ## num antennas
    cf = 2.412e9  ## center freq for uplink
    bw = 20e6  ## bandwidth over which channel is observed
    l1 = get_lambsWifi(cf, bw, 64)  ## wavelengths, lambda for subcarriers in channel
    d_sep = None  ## for generating channels, what is the min separation between any two components
    n_cores = 4
    num_chans = 100000
    cf2 = 2.437e9 ## center freq for downlink
    l2 = get_lambsWifi(cf2, bw, 64)
```
