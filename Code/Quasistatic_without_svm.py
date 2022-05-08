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

import ifxdaq
import processing
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import fft

config_file = "radar_configs/RadarIfxBGT60.json"
raw_data    = []

with RadarIfxAvian(config_file) as device:                             # Initialize the radar with configurations
    
    for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
        
        raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
        
        if(len(raw_data) > 1999 and len(raw_data) % 2000 == 0):        # 5000 is the number of frames. which corresponds to 5seconds
            
            data = np.swapaxes(np.asarray(raw_data), 0, 1)
            
            st_data = remove_mean(data)
          
            x_rdi = fft.doppler_fft(data)

            x=np.mean(x_rdi,axis=0)
            y=np.mean(x,axis=1)
            z = abs(y)
            r=z*(10e12)

            sum= np.sum(r[0:2750])
            print(sum)
            if(sum>14):
                print("moving")
            else:
                print("not moving")
           
            raw_data =[]
