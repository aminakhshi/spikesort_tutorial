#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different Filter designs to filter a data series with FIR coefficients.
Please cite github for bug reports and usage

Examples is given from scipy docs

@author: Amin Akhshi
Email: amin.akhshi@gmail.com
Github: https://aminakhshi.github.io
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
rng = np.random.default_rng()


def roll(data, shift):
    """shift vector
    """
    new_arr = np.zeros(len(data), dtype=data.dtype)

    if shift < 0:
        shift = shift - len(data) * (shift // len(data))
    if shift == 0:
        return
    new_arr[0:shift] = data[len(data)-shift: len(data)]
    new_arr[shift:len(data)] = data[0:len(data)-shift]
    return new_arr


def fir_filt(data, coeff):
    """
    Filter the data series with a set of FIR coefficients
    :param data: a numpy array or pandas series
    :param coeff: FIR coefficients. Should be with odd length and sysmmetric.
    :return: The filtered data series
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # apply the filter
    data_filtered = signal.lfilter(coeff, 1.0, data)
    # data_filtered = pd.Series(data_filtered)

    # reverse the time shift caused by the filter,
    # corruption regions contain zeros
    # If the number of filter coefficients is odd, the central point *should*
    # be included in the output so we only zero out a region of len(coeff) - 1
    data_filtered[:(len(coeff) // 2) * 2] = 0
    data_filtered = roll(data_filtered, -len(coeff) // 2)
    return data_filtered


def lpass(data, fs, f_cut, order, beta=5.0):
    """
    FIR lowpass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The low frequency cutoff treshhiold
    :param order:   The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k = f_cut / nyqst
    coeff = signal.firwin(order * 2 + 1, k, window=('kaiser', beta))
    return fir_filt(data, coeff)


def hpass(data, fs, f_cut, order, beta=5.0):
    """
    FIR hpass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The high frequency cutoff treshhiold
    :param order:   The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k = f_cut / nyqst
    coeff = signal.firwin(
        order * 2 + 1, k, window=('kaiser', beta), pass_zero=False)
    return fir_filt(data, coeff)


def bpass(data, fs, cut_low, cut_high, order, beta=5.0):
    """
    FIR band pass filter design
    :param data: a numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param cut_low: The low frequency cutoff treshhiold
    :param cut_high: The high frequency cutoff treshhiold
    :param order: The number of corrupted samples on each  side of the data
    :param beta: side lobe attenuation parameter in kaiser window
    :return:
    """
    nyqst = float(fs/2)
    k1 = cut_low / nyqst
    k2 = cut_high / nyqst
    coeff = signal.firwin(order * 2 + 1, [k1, k2], window=('kaiser', beta))
    return fir_filt(data, coeff)


def notch_filt(data, fs, f_cut, Q=30):
    """
    IIR notch filter design
    :param data: numpy array or pandas series
    :param fs: sampling frequency of the data array
    :param f_cut: The cutoff frequency in Hz
    :param Q: Order of the quality factor filter
    :return:
    """
    nyqst = float(fs/2)
    K = f_cut / nyqst  # Normalized Frequency
    b, a = signal.iirnotch(K, Q)
    data_filtered = signal.lfilter(b, a, data)
    return data_filtered

# =============================================================================
# Place your data array here and comment this example block 
# For each filter follow the rule in examples
# =============================================================================
"""
Generating a random data
"""
fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 45
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
data = amp*np.sin(2*np.pi*freq*time)
data += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

"""
Applying filters
Note = Place your data array insted
""""
lpass_data = lpass(data, fs, f_cut=80, order=512)
hpass_data = hpass(data, fs, f_cut=60, order=512)
bpass_data = bpass(data, fs, cut_low=60, cut_high=100, order=512)
notch_data = notch_filt(data, fs, freq)

series = {'raw': data,
          'lpass': lpass_data,
          'hpass': hpass_data,
          'bpass': bpass_data,
          'notch': notch_data}

fig, ax = plt.subplots(figsize=(9, 6))
for key, val in series.items():
    freqs, psd = signal.welch(val, fs, nperseg=1024)
    ax.loglog(freqs, psd, label=key)
    ax.set_xlim([10, 500])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.legend()
plt.show()
