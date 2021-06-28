import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve
import random
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import librosa
import scipy.signal
import stft_test

def main():
    
    win_size=512
    hop_size=256
    fs=48000
    wav, wav_azi, white, white_azi=dataloader()  
    delta=0.04
    
    num_angle=720    
    theta_d=90

    beamforming_signal(wav, win_size, hop_size, fs, theta_d, delta)
    for fs in range(800):
        steer_set=[]
        angle_set=[]
        f=(fs)*10
        f=2000

        steering_vector_d=calc_steer(wav.shape[1], theta_d, f, delta)
        h=steering_vector_d
        beamforming_signal(wav, h)
        for i in range(num_angle):
            azi=i*360/num_angle
            
            steering_vector=calc_steer(wav.shape[1], azi, f, delta)
            
            beampattern=np.conj(steering_vector).T@h
            beampattern=np.abs(beampattern)
            # beampattern/=wav.shape[1]**2
            beampattern=np.real(beampattern)
            
            # beampattern=np.real(beampattern*np.conj(beampattern))
            
            steer_set.append(beampattern)
            angle_set.append(azi*np.pi/180)
        # print(azi)
        
        # plt.clf()
        steer_set=np.array(steer_set)
        steer_set=20*np.log10(np.abs(steer_set)/np.max(steer_set))
        angle_set=angle_set
        # steer_set=steer_set
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        ax.set_title(str(f)+'   '+str(theta_d))
        ax.set_yticks([-20, -10,0])
        ax.set_yticklabels(['-20dB', '-10dB', '0dB'])
        ax.plot(angle_set, steer_set)
        plt.show(block=True)
        # plt.pause(0.3)
        # plt.close()
        exit()
        # ax.cla()
        
    

    # steering_vector=calc_steer(wav.shape[1], wav_azi, 1000, delta)



def beamforming_signal(wav, win_size, hop_size, fs, theta_d, delta):
    original_wav_shape=wav.shape[0]
    no_ch=wav.shape[1]
    n_iter=(wav.shape[0])//hop_size
    pad_size=win_size+(n_iter-1)*hop_size-original_wav_shape
   
    wav=np.pad(wav, ((0, pad_size), (0,0)))
    
    freq_bin=np.linspace(0, fs//2, win_size//2+1)
    
    h=np.zeros((freq_bin.shape[0], no_ch), dtype=np.complex64)  
    for num, f in enumerate(freq_bin):
        steering_vector_d=calc_steer(no_ch, theta_d, f, delta)
        h[num,:]=steering_vector_d/no_ch
    
    h=np.conj(h)   

    result=np.zeros(wav.shape[0])    
    window_function=np.hanning(win_size).reshape(-1,1)
    # window_function=np.ones(win_size).reshape(-1,1)



    for fram_no in range(n_iter):
        fram=wav[fram_no*hop_size:fram_no*hop_size+win_size,:]*window_function
        fram=fft(fram, axis=0)[:win_size//2+1,:]
        fram=(h*fram).sum(-1)
        flipped=np.flip(np.conj(fram[1:-1]))
        fram=np.concatenate((fram, flipped))        
        fram=np.real(ifft(fram))#*np.squeeze(window_function, axis=-1)
        # print(fram.shape, result.shape)
        result[fram_no*hop_size:fram_no*hop_size+win_size]=fram+result[fram_no*hop_size:fram_no*hop_size+win_size]
    result=result[:original_wav_shape]
    sf.write('beamform5.wav', result, fs)
    plt.plot(result)
    plt.show()
    exit()
    result=np.zeros(wav_freq.shape[0], dtype=np.complex64)    
    delta=0.04

    for num, f in enumerate(freq_bin):
        steering_vector_d=calc_steer(wav_freq.shape[1], 0, f, delta)
        h=steering_vector_d
        a=np.conj(h).T@wav_freq[num,:]
        result[num]=a
    
    flipped=np.conj(np.flip(result[1:-1]))
    
    result=np.concatenate((result,np.conj(flipped)))
    
    result=ifft(result)



def calc_steer(vec_size, wav_azi, f, delta):
    c=340
    steering_vector=np.ones(vec_size, dtype=np.complex64)

    for i in range(0, steering_vector.shape[0]):
        formula=-(i)*2*np.pi*f*delta/c*np.cos(wav_azi/180*np.pi)
        steering_vector[i]=np.exp(formula*1j)
        # print(steering_vector)
        # exit()
        

    return steering_vector
    

def dataloader():
    wav='../wav_data/p225_004.wav'
    rir='../wav_data/linear_n8_az90_el0_r1.5.npz'
    wav,fs=sf.read(wav)
    rir=np.load(rir, allow_pickle=True)
    rir=rir['rir']

    final=None
    for num in range(rir.shape[1]):
        result=oaconvolve(wav, rir[:,num])
        result=np.expand_dims(result, axis=-1)
        if num==0:
            final=result
        else:
        
            final=np.concatenate((final, result), axis=1)
    wav=final
    
    white=np.random.normal(0, 0.02, size=final.shape[0])
    rir='../wav_data/linear_n8_az180_el0_r1.5.npz'
    rir=np.load(rir, allow_pickle=True)
    rir=rir['rir']
    final=None
    for num in range(rir.shape[1]):
        result=oaconvolve(white, rir[:,num])
        result=np.expand_dims(result, axis=-1)
        if num==0:
            final=result
        else:
        
            final=np.concatenate((final, result), axis=1)
    white=final
    wav+=white[:wav.shape[0],:]
    # sf.write('noisy.wav', wav, fs)
    return wav, 0, final, 180


if __name__=='__main__':
    random.seed(0)
    np.random.seed(0)
    main()