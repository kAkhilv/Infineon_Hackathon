import ifxdaq
import processing
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import numpy as np
import scipy
from scipy import signal
from ellipse import LsqEllipse
import circle_fit as cf
#import matplotlib.pyplot as plt
from fft import range_fft
import pandas as pd
#import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fft

config_file = "radar_configs/RadarIfxBGT60.json"
raw_data    = []

cutOff_lp = 249/500 #cutoff frequency in rad/s
fs_lp = 5 #sampling frequency in rad/s
order_lp = 6

cutOff_hp = 249/500 #cutoff frequency in rad/s
fs_hp = 5 #sampling frequency in rad/s
order_hp = 6

def remove_mean(raw_data):
    raw_data -= np.mean(raw_data, axis=-1, keepdims=True)
    raw_data -= np.mean(raw_data, axis=-2, keepdims=True)
    return raw_data

def get_range_index_recordwise(raw_data):
    range_window_func = lambda x: scipy.signal.windows.chebwin(x, at=100)
    range_data  = range_fft(raw_data, range_window_func)

    range_tmp = np.sum(np.sum(np.abs(range_data), axis=-2), axis=0)
    #print("hello")
    range_idx = np.argmax(range_tmp, axis=-1)


    return range_idx, range_data

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)
def butter_lowpass_filter(data, cutOff, fs, order=5):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutOff_lp = 3/500 #cutoff frequency in rad/s
fs_lp = 1 #sampling frequency in rad/s
order_lp = 4
#Fs=1, lowcut = 0.2/500 , highcut = 0.5/500, order=6 for BR

cutOff_lp2 = 50/500 #cutoff frequency in rad/s
fs_lp2 = 1 #sampling frequency in rad/s
order_lp2 = 4

cutOff_hp = 0.8/500 #cutoff frequency in rad/s
fs_hp = 1 #sampling frequency in rad/s
order_hp = 4

lowcut = 0.2/500
highcut = 2/500
order_bp = 4
fs_bp = 1

with RadarIfxAvian(config_file) as device:                             # Initialize the radar with configurations
    
    for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
        
        raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
        
        if(len(raw_data) > 9999 and len(raw_data) % 10000 == 0):        # 5000 is the number of frames. which corresponds to 5seconds
            
            data = np.swapaxes(np.asarray(raw_data), 0, 1)                     
            #bp = butter_bandpass_filter(data, lowcut, highcut, fs_bp, order_bp)
            lp2= butter_lowpass_filter(data, cutOff_lp2, fs_lp2, order_lp2)
            phases, abses, _, _ = processing.do_processing(lp2)       # preprocessing to get the phase information
            
            phases              = np.mean(phases, axis=0)
            
            hp = butter_highpass_filter(phases, cutOff_hp, fs_hp,order_hp)
            lp= butter_lowpass_filter(hp, cutOff_lp, fs_lp, order_lp)            
            mean = np.mean(lp)
            lp = lp - mean
            plt.plot(lp)
            plt.show()
            index = scipy.signal.find_peaks(lp)
            count=0
            for i in index[0]:
                if lp[i] > 0.17:
                    count = count +1
            print(count*6)
            del data
            del phases
            raw_data = []
            
