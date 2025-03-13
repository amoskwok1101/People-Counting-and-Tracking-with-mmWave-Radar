import numpy as np
import scipy.io as sio

def data(path):
    #organize raw adc Data into proper dimension
    mat = sio.loadmat(path)['adcData']
    frame_0=[]
    
    for j in range(128):
        chirp=[]
        for c in range(4):
            chirp.append(mat[c,256*j:256*(j+1)])
        frame_0.append(chirp)
    
    #(num_chirps_per_frame, num_rx_antennas, num_adc_samples)
    return np.array(frame_0)
        
            




    
