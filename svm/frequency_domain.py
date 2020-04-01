#Frequency Domain Feature Extraction
from scipy import signal
from load_data import *
from matplotlib import pyplot as plt
import numpy as np

""" WAVELET WORKS BETTER, CAPTURES TEMPORAL CHANGES
def STFT(data):
    for d in data:
        f, t, Zxx = signal.stft(d, fs=1.0, window='hann', nperseg=256, 
            noverlap=None, nfft=None, detrend=False, return_onesided=True, 
                boundary='zeros', padded=True, axis=-1)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    return
"""
def mag(data):
    for d in data:
        plt.plot(d)
        plt.show()  

    
def Wavelet(data):
    answer = []
    for d in data:
        widths = np.arange(1, RESOLUTION)
        cwtmatr = signal.cwt(d, signal.ricker, widths)
        answer.append(cwtmatr)
    return answer
        
    
def plot_wavlet(data):
        results = Wavelet(data)
        for mats in results:
            plt.imshow(mats, extent=[-1, 1, 1, RESOLUTION], cmap='PRGn', aspect='auto',
            vmax=abs(mats).max(), vmin=-abs(mats).max())
            plt.show()    


def add_freq_features(data):
    W = Wavelet(data)
    result = []
    for i in range(len(data)):
        d = data[i]
        w = W[i].flatten()
        print(d.shape,w.shape)
        entry = np.ndarray.tolist(np.append(d,w))
        print(len(entry))
        result.append(entry)
    return result

def plot_all_features(data):
    mag(data)
    plot_wavlet(data)    
    #STFT(data)
    
#PLOTS GRAPHS OF FREQUENCIES
#plot_all_features(forward_fall)
#add_freq_features(forward_fall)
