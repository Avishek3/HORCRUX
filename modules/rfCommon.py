import numpy as np
import random

def get_poll(box, dist):
    for i in range(50):
        if box[i]==2:
            box[i]==1
    for x in dist:
        idx = int(x % 50)
        for i in range(-2,3):
            if(idx+i>=0 and idx+i<400):
                box[idx+i]+=1

    return box






def get_X(pds, divs):
    mat = []
    for d in pds:
        s = np.exp(-2j * np.pi * d / divs)
        mat.append(s)
    mat = np.array(mat).transpose()
    return mat


def get_a(params, distances):
    ds = params[0]
    amps = params[1]
    a = np.zeros(len(distances) + 1)
    for i in range(len(amps)):
        d = ds[i]
        amp = amps[i]
        pos = np.digitize(d, distances)
        a[pos] = amp
    return a


def get_d_ir(chan, distances, l):
    nds = len(distances)
    mat = np.zeros([nds]).astype(complex)
    for i in range(nds):
        d = distances[i]
        temp = np.exp(2j * np.pi * d / l)
        temp = np.mean(temp * chan)
        mat[i] = temp
    return mat


def get_peaks(sig):
    s = []
    s.append(sig[1])
    s.extend(sig)
    s.append(sig[-2])
    s = np.array(s)
    l = s[1:-1] - s[2:]
    l = np.where(l < 0, -1.0, 1.0)
    l += s[1:-1] - s[:-2]
    pos = np.where(l > 1)[0]
    l = np.zeros(len(s))
    l[pos] = s[pos + 1]
    l = l[1:-1]
    return l


def get_peak_distances(sig, distances):
    peaks = get_peaks(sig)
    pos = np.where(peaks != 0)
    ds = distances[pos]
    return ds


def get_lambs(cf, bw, nfft):
    c = 3e8
    f = np.fft.fftfreq(nfft)
    f = f * bw
    cf = f + cf
    l1 = []
    for f in cf:
        l1.append(c / f)
    return np.array(l1)


def get_synth_params_sep(n_paths, lo, hi, sep=None):
    '''
    using random values for the 4-tuples for
    each path. "sep" is the minimum distance between components.
    '''
    if sep == None:
        return get_synth_params(n_paths, lo, hi)

    np.random.seed()
    div = 10000.0
    hi = hi * div
    lo = lo * div
    population = np.arange(lo, hi, 2 * sep * div)
    #print(len(population))
    assert len(population) >= n_paths, " assert error: get_synth_params_sep(...): population size smaller than n_paths"
    d_ns = np.array(random.sample(list(population), n_paths))
    #print(d_ns)
    fractional = np.random.rand(n_paths) * sep * div
    d_ns = d_ns + fractional
    #print(d_ns)
    d_ns = d_ns / div
    #print(d_ns)
    d_ns = d_ns[d_ns <= hi / div]
    #print(d_ns)
    if n_paths > 1:
        assert max(d_ns) - min(d_ns) >= sep, " assert error: get_synth_params_sep(...): min separation assertion failed"

    a_ns = np.random.rand(n_paths)
    a_ns = a_ns / np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    psi_ns = np.cos(np.random.rand(n_paths) * np.pi)  ### 0 to pi range
    
    return d_ns, a_ns, phi_ns, psi_ns


def get_synth_params_sepAOA(n_paths, lo, hi, sep=None):
    '''
    using random values for the 4-tuples for
    each path. "sep" is the minimum distance between components.
    '''
    if sep == None:
        return get_synth_params(n_paths, lo, hi)

    np.random.seed()
    div = 10000.0
    hi = hi * div
    lo = lo * div
    population = np.arange(lo, hi, 2 * sep * div)
    #print(len(population))
    assert len(population) >= n_paths, " assert error: get_synth_params_sep(...): population size smaller than n_paths"
    d_ns = np.array(random.sample(list(population), n_paths))
    #print(d_ns)
    fractional = np.random.rand(n_paths) * sep * div
    d_ns = d_ns + fractional
    #print(d_ns)
    d_ns = d_ns / div
    #print(d_ns)
    d_ns = d_ns[d_ns <= hi / div]
    #print(d_ns)
    if n_paths > 1:
        assert max(d_ns) - min(d_ns) >= sep, " assert error: get_synth_params_sep(...): min separation assertion failed"

    a_ns = np.random.rand(n_paths)
    a_ns = a_ns / np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    

    psi_ns = np.cos(np.random.rand(n_paths) * np.pi)  ### 0 to pi range

    psi_angles = np.arccos(psi_ns)*180/np.pi
    
    return d_ns, a_ns, phi_ns, psi_ns, psi_angles


def get_synth_params(n_paths, lo, hi):
    '''
    using random values for the 4-tuples for
    each path
    '''
    np.random.seed()
    div = 10000.0
    hi = hi * div
    lo = lo * div
    d_ns = np.random.randint(lo, hi, n_paths).astype(float) / div  ### to add a fractional part

    a_ns = np.random.rand(n_paths)
    if n_paths>1:
        a_ns = a_ns / np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    psi_ns = np.cos(np.random.rand(n_paths) * np.pi)  ### 0 to pi range

    
    

    return d_ns, a_ns, phi_ns, psi_ns


def get_synth_paramsAOA(n_paths, lo, hi):
    '''
    using random values for the 4-tuples for
    each path
    '''
    np.random.seed()
    div = 10000.0
    hi = hi * div
    lo = lo * div
    d_ns = np.random.randint(lo, hi, n_paths).astype(float) / div  ### to add a fractional part

    a_ns = np.random.rand(n_paths)
    if n_paths>1:
        a_ns = a_ns / np.sum(a_ns)

    phi_ns = np.random.rand(n_paths)
    psi_ns = np.cos(np.random.rand(n_paths) * np.pi)  ### 0 to pi range
    
    psi_angles = np.arccos(psi_ns)*180/np.pi
    return d_ns, a_ns, phi_ns, psi_ns, psi_angles

def get_chans_from_params(params, K, sep, lambs):
    '''
    given the params/4-tuples for the physical paths, compute the channel
    across K antenna for given wavelengths.
    Based on Equation 3 of paper
    '''
    d_ns, a_ns, phi_ns, psi_ns = params
    N = len(d_ns)
    I = len(lambs)
    H = np.zeros([I, K]).astype(complex)

    for i_wl in range(len(lambs)):
        wl = lambs[i_wl]
        ### based on eq 3
        for i_K in range(K):  ###for each antenna
            t = 0
            for i_N in range(N):  ###for each path
                c1 = a_ns[i_N] * np.exp((-2j * np.pi * d_ns[i_N] / wl) + 1j * phi_ns[i_N])
                c1 = c1 * np.exp(-2j * np.pi * (i_K) * sep * psi_ns[i_N] / wl)
                H[i_wl, i_K] += c1
    return H

def get_chans_from_paramsAOA(params, K, sep, lambs):
    '''
    given the params/4-tuples for the physical paths, compute the channel
    across K antenna for given wavelengths.
    Based on Equation 3 of paper
    '''
    d_ns, a_ns, phi_ns, psi_ns,x = params
    N = len(d_ns)
    I = len(lambs)
    H = np.zeros([I, K]).astype(complex)

    for i_wl in range(len(lambs)):
        wl = lambs[i_wl]
        ### based on eq 3
        for i_K in range(K):  ###for each antenna
            t = 0
            for i_N in range(N):  ###for each path
                c1 = a_ns[i_N] * np.exp((-2j * np.pi * d_ns[i_N] / wl) + 1j * phi_ns[i_N])
                c1 = c1 * np.exp(-2j * np.pi * (i_K) * sep * psi_ns[i_N] / wl)
                H[i_wl, i_K] += c1
    return H

def add_noise_snr_range(chans, min_snr, max_snr):
    n = chans.shape[0]
    r, c = chans.shape
    snrs = np.random.randint(min_snr, max_snr + 1, n)
    scale = np.power(10, -1 * snrs / 20.0)
    scale = scale.reshape([-1, 1])
    noise = np.random.randn(r, c) + 1j * np.random.randn(r, c)
    noise = noise / np.abs(noise)
    noise = scale * noise
    chans = chans + noise
    return chans


def add_noise_matrix(data, snr):
    sig_amp = np.mean(np.abs(data))
    noise_amp = sig_amp * np.power(10, (-1 * snr / 20.0))
    r, c = data.shape
    noise = np.random.randn(r, c) + 1j * np.random.randn(r, c)
    noise = noise / np.abs(noise)
    noise = noise * noise_amp
    noisy_signal = data + noise
    return noisy_signal


def get_beam_power(ch_act, ch_pred):
    '''
    ch: IxK, I=#subcarriers, K=#antennae
    '''
    x = ch_act / ch_pred
    x = np.sum(x, 1)  ##x = value in each subcarrier = sum across all antennae
    x_amp = np.abs(x)  ##get abs for each subcarrier
    x_pwr = x_amp ** 2  ##power
    x_mean_pwr = np.mean(x_pwr)  ##mean power

    return x_mean_pwr


def beam_eval(obs_f2, pred_f2, noise_pwr, snr):
    '''
    ch: IxK, I=#subcarriers, K=#antennae
    '''
    avg_sig_amp = np.mean(np.abs(obs_f2))
    scale = 10 ** (snr / 20.0)
    scale = scale / avg_sig_amp
    scale = scale * np.sqrt(noise_pwr)

    obs_f2 = obs_f2 * scale
    pred_f2 = pred_f2 / np.abs(pred_f2)

    r, c = obs_f2.shape
    rand = np.ones([r, c]) + 1j * np.zeros([r, c])
    rand = np.random.randn(r, c) + 1j * np.random.randn(r, c)
    rand = rand / np.abs(rand)

    a = get_beam_power(obs_f2, pred_f2)
    b = get_beam_power(obs_f2, rand)
    c = get_beam_power(obs_f2, obs_f2 / np.abs(obs_f2))

    a = 10 * np.log10(a / noise_pwr)
    b = 10 * np.log10(b / noise_pwr)
    c = 10 * np.log10(c / noise_pwr)

    a = round(a, 2)
    b = round(b, 2)
    c = round(c, 2)

    return a, b, c

def FIRE_beam_eval(H_org_data,H_pred_data,K) :
    nfft = 26
    # H_org_data = H_org_data.reshape(1,nfft*K)
    # H_pred_data = H_pred_data.reshape(1,nfft*K)

    d1 = np.mean(np.abs(np.multiply(np.conjugate(H_org_data),H_pred_data)))
    d2 = np.mean(np.abs(H_pred_data[0:nfft]))
    snr = d1/(K*d2)


    return 20*np.log10(snr/0.01)