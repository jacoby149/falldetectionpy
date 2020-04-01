#Frequency Domain Feature Extraction
from scipy import signal
from load_data import *
from matplotlib import pyplot as plt
import numpy as np

col = "magnitude"
forward_fall = load_data_from_csv("./data/forward_fall.csv",col)
side_fall = load_data_from_csv("./data/s_fall.csv",col)
walking = load_data_from_csv("./data/walking.csv",col)
jumping = load_data_from_csv("./data/jumping.csv",col)



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
    
def Wavelet(data):
    for d in data:
        widths = np.arange(1, 31)
        cwtmatr = signal.cwt(d, signal.ricker, widths)
        plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()    
    return

Wavelet(forward_fall)    
STFT(forward_fall)
