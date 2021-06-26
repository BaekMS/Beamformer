import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve
import random
import matplotlib.pyplot as plt


def main():
    wav, wav_azi, white, white_azi=dataloader()
    delta=0.03
    # first position is heading azimuth zero, plot in left side
    
    num_angle=720
  
    for fs in range(800):
        steer_set=[]
        angle_set=[]
        f=(fs)*10
        for i in range(num_angle):
            azi=i*360/num_angle
            steering_vector=calc_steer(wav.shape[1], azi, f, delta)
            h=steering_vector/8
            beampattern=steering_vector.T@h
            
            beampattern=np.real(beampattern*np.conj(beampattern))
            # print(beampattern)
            # exit()
            steer_set.append(beampattern)
            angle_set.append(azi*np.pi/180)

        # plt.clf()
        angle_set=angle_set[:180]
        steer_set=steer_set[:180]
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title(str(f))
        ax.plot(angle_set, steer_set)
        plt.show(block=False)
        plt.pause(0.3)
        plt.close()
        # ax.cla()
        
    

    # steering_vector=calc_steer(wav.shape[1], wav_azi, 1000, delta)
    

def calc_steer(vec_size, wav_azi, f, delta):
    c=340
    steering_vector=np.ones(vec_size, dtype=np.complex64)

    for i in range(1, steering_vector.shape[0]):
        formula=-(i)*2*np.pi*f*delta/340*np.cos(wav_azi/180*np.pi)
        steering_vector[i]=np.exp(formula*1j)
        

    return steering_vector
    

def dataloader():
    wav='../wav_data/p225_004.wav'
    rir='../wav_data/linear_n8_az0_el0_r1.5.npz'
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

    white=np.random.normal(0, 0.02, size=final.shape[-1])
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
    return wav, 0, final, 180


if __name__=='__main__':
    random.seed(0)
    np.random.seed(0)
    main()