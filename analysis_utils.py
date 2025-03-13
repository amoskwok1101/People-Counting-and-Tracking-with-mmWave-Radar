import pandas as pd
import numpy as np
import pickle
import scipy.io as sio
from scipy import fftpack
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import mmwave
from sklearn.cluster import DBSCAN
from scipy import signal
range_res, bandwidth = mmwave.dsp.range_resolution(256,10000,29.9817)
NUM_RX = 4
# VIRT ANT = NUM_RX * NUM_TX = 4 * 2 = 8
VIRT_ANT = 8
ANGLE_RES = 1
ANGLE_RANGE = 90
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 127
RANGE_RESOLUTION = range_res

def range_angle_heatmap(frame):
    def blackman(signal_):
        # reduce side lobes
        return np.convolve(signal_, signal.windows.blackmanharris(35), mode='same')
    def simple_moving_average(signal, window=5):
        return np.convolve(signal, np.ones(window)/window, mode='same')
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            frame[i,j] = blackman(frame[i,j])
            #frame[i,j] = simple_moving_average(frame[i,j],3)
    
    
    cube = mmwave.dsp.range_processing(frame)
    #cube = (num_chirps_per_frame, num_rx_antennas, num_range_bins)
    range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    num_vec, steering_vec = mmwave.dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)
    
    #static clutter removal
    mean = cube.mean(0) #mean for each chirp
    cube = cube - mean
    
    

    
    
    def simple_moving_average(signal, window=5):
        return np.convolve(signal, np.ones(window)/window, mode='same')
    
    
    cube_filtered = np.empty((cube.shape[0],cube.shape[1],127),dtype=np.complex_)
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            freq = fftpack.fftfreq(len(cube[i,j,:])) * 10000
            cube_filtered[i,j,:] = cube[i,j,:][freq > 0]
            #cube_filtered[i,j,:] = blackman(cube_filtered[i,j,:])
            cube_filtered[i,j,:] = simple_moving_average(cube_filtered[i,j,:],window=3)
            #cube[i,j,:] = gaussian_filter1d(cube[i,j,:],sigma=2)
    
           
    # we referenced PreSense Team mmWave implementation for Capon beamforming implementation
    # https://github.com/PreSenseRadar/OpenRadar
    beamWeights   = np.zeros((VIRT_ANT, BINS_PROCESSED), dtype=np.complex_)
    cube_filtered = np.concatenate((cube_filtered[0::2, ...], cube_filtered[1::2, ...]), axis=1)
    # split into even odd chirp

    # perform capon beamforming to determine angle of arrival
    
    for i in range(BINS_PROCESSED):
        range_azimuth[:,i], beamWeights[:,i] = mmwave.dsp.aoa_capon(cube_filtered[:, :, i].T, steering_vec, magnitude=True)

    #heatmap_log = np.log2(range_azimuth)

    return range_azimuth

def peak_finding_cfar(heatmap,l_bound,guard_len=4,noise_len=16,scale_a=1.5,scale_r=1.6):

    # cfar in azimuth direction
    # we referenced PreSense Team mmWave implementation for CFAR implementation
    # https://github.com/PreSenseRadar/OpenRadar
    first_pass, _ = np.apply_along_axis(func1d=mmwave.dsp.os_,
                                            axis=0,
                                            arr=heatmap,
                                            k=noise_len,
                                            guard_len=guard_len,
                                            noise_len=noise_len,
                                            scale = scale_a)
        
    # cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=mmwave.dsp.os_,
                                                    axis=0,
                                                    arr=heatmap.T,
                                                    k=noise_len,
                                                    guard_len=guard_len,
                                                    noise_len=noise_len,
                                                    scale = scale_r)

    SKIP_SIZE = 4
    noise_floor = noise_floor.T
    first_pass = (heatmap > first_pass)
    second_pass = (heatmap > second_pass.T)
    peaks = (first_pass & second_pass)
    #peaks = first_pass

    #Ignore boundary response
    peaks[:SKIP_SIZE, :] = 0
    peaks[-SKIP_SIZE:, :] = 0
    peaks[:, :SKIP_SIZE] = 0
    peaks[:, -SKIP_SIZE:] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    
    ranges = ranges * RANGE_RESOLUTION
    azimuths = (azimuths - (ANGLE_BINS // 2)) *np.pi/180
    
    return ranges,azimuths, pairs

def peak_finding(heatmap,coeff):
    #peak finding based on fixed threshold
    threshold = np.mean(heatmap) + coeff*np.std(heatmap)
    peaks = heatmap > threshold
    SKIP_SIZE = 4

    peaks[:SKIP_SIZE, :] = 0
    peaks[-SKIP_SIZE:, :] = 0
    peaks[:, :SKIP_SIZE] = 0
    peaks[:, -SKIP_SIZE:] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    
    ranges = ranges * RANGE_RESOLUTION
    azimuths = (azimuths - (ANGLE_BINS // 2)) *np.pi/180
    
    return ranges,azimuths,pairs



    

def coord(ranges,azimuths):
    #transform range-azimuth plane to x-y plane
    coor = []
    for i in range(len(ranges)):
        r = ranges[i]
        a = azimuths[i]
        x = r*np.sin(a)
        y = r*np.cos(a)
        coor.append([x,y])
    return np.array(coor)



