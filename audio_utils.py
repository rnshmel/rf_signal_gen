import numpy as np
import sys
import wave
from scipy import signal
import math
import mod_utils as mu
#import matplotlib.pyplot as plt

# globals
ch_num = 0
samp_width = 0
frame_num = 0
frame_rate = 0

# audio AM modulation
# takes a single tone carrier and modulates with AM
def audio_am_mod(data, threshold, N):
    # step 1: generate the carrier wave
    error_code, carrier = mu.tone_gen(0, len(data), N)
    # now scale the data based on mod threshold
    data = (data + 1)/np.max(data+1)
    data = (data*threshold)+(1-threshold)
    # apply modulation
    signal = (carrier * data)
    return signal

def audio_interp_hold(y_data, I_val):
    interp_data = np.zeros(len(y_data)*I_val, dtype="float32")
    for i in range(0, len(y_data)):
        for j in range(0, I_val):
            interp_data[(i*I_val)+j] = y_data[i]
    return interp_data

# audio interpolation
# interp by zero padding between samples
def audio_interp_zeros(data, scale):
    interp_array = np.zeros(len(data)*scale, dtype="complex64")
    marker = 0
    for i in range(0, len(data)):
        interp_array[marker] = data[i]
        marker = marker + scale
    return interp_array

# audio interpolation
# decimate by removing 1-in-N samples
def audio_decimate(data, scale):
    dec_array = np.zeros(int(len(data)/scale), dtype="complex64")
    marker = 0
    for i in range(0, len(dec_array)):
        dec_array[i] = data[marker]
        marker = marker + scale
    return dec_array

# low pass audio filter
def audio_filter(data, cutoff, stop, samp_rate):
    iir_sos_lpf = signal.iirdesign(cutoff, stop, 1, 100, output="sos", fs=samp_rate)
    filtered_data = signal.sosfilt(iir_sos_lpf, data)
    return filtered_data.astype("complex64")

# perform analog FM mod on data
# assume data is +1 to -1 floating point
def audio_fm_mod(data, freq_div, fs):
    raw_array = (data * freq_div/2)
    # integrate frequency to calculate the phase at any given sample value
    phi_array = np.cumsum(raw_array*2*np.pi)/fs
    # get i values with cosine
    cos_array = np.cos(phi_array)
    # get j values with sine
    sin_array = np.sin(phi_array)
    # combine into a complex array
    # save as complex64 (default is complex128)
    complex_array = (cos_array + 1j*sin_array).astype("complex64")
    return complex_array

# loads a wav file
# file can be 8 or 16 bit audio - mono or stereo
# returns the file in a np array (dtype = float32)
def wav_file_load(filename):
    global ch_num
    global samp_width
    global frame_num
    global frame_rate
    # open wav file - read
    wav_obj_r = wave.open(filename, mode='rb')
    # get basic params
    ch_num = int(wav_obj_r.getnchannels())
    samp_width = int(wav_obj_r.getsampwidth())
    frame_num = int(wav_obj_r.getnframes())
    frame_rate = int(wav_obj_r.getframerate())

    # read byte data into bytes object
    wav_data_raw = wav_obj_r.readframes(frame_num)

    # depending on sample width, the data is either
    # 1) uint 8 
    # 2) signed int 16
    # convert to float
    if (samp_width == 1):
        # uint8
        if ch_num == 1:
            # mono
            wav_data = np.zeros(frame_num, dtype="uint8")
            for i in range(0, len(wav_data_raw)):
                wav_data[i] = wav_data_raw[i]
            wav_data_float = wav_data.astype("float32") - 127
            wav_data_float = wav_data_float/255
            wav_data_float = wav_data_float/np.max(abs(wav_data_float))
        else:
            # stereo
            count = 0
            wav_data_r = np.zeros(frame_num, dtype="uint8")
            wav_data_l = np.zeros(frame_num, dtype="uint8")
            for i in range(0, len(wav_data_raw), 2):
                wav_data_r[count] = wav_data_raw[i]
                wav_data_l[count] = wav_data_raw[i+1]
                count = count + 1
            wav_data_r = wav_data_r.astype("float32") - 127
            wav_data_l = wav_data_l.astype("float32") - 127
            wav_data_r = wav_data_r/255
            wav_data_l = wav_data_l/255
            wav_data_float = (wav_data_r/2.0)+(wav_data_l/2.0)
            wav_data_float = wav_data_float/np.max(abs(wav_data_float))
    else:
        # int16
        # check if mono or stereo
        if ch_num == 1:
            # mono
            wav_data = np.zeros(frame_num, dtype="int16")
            count = 0
            for i in range(0, len(wav_data_raw), 2):
                sample = (wav_data_raw[i+1] << 8) | wav_data_raw[i]
                wav_data[count] = sample
                count = count + 1
            wav_data_float = wav_data.astype("float32")/np.max(abs(wav_data))
        else:
            # stereo
            wav_data_r = np.zeros(frame_num, dtype="int16")
            wav_data_l = np.zeros(frame_num, dtype="int16")
            count = 0
            for i in range(0, len(wav_data_raw), 4):
                sample_r = (wav_data_raw[i+1] << 8) | wav_data_raw[i]
                sample_l = (wav_data_raw[i+3] << 8) | wav_data_raw[i+2]
                wav_data_r[count] = sample_r
                wav_data_l[count] = sample_l
                count = count + 1
            wav_data_r = wav_data_r.astype("float32")
            wav_data_l = wav_data_l.astype("float32")
            wav_data_float = (wav_data_r/2.0)+(wav_data_l/2.0)
            wav_data_float = wav_data_float/np.max(abs(wav_data_float))
    return wav_data_float

def fm_analog_mod(filename, mod_bw, samp_rate, scale):
    try:
        # step 1: open the file
        init_data = wav_file_load(filename)
        # step 2: run the inital interpolation using interp-and-hold
        # this interpolates to a value for quadrature fm modulation
        # this step only needs to be done if your quadrature rate is greater
        # than 2x the wave file frame rate
        interp_factor = 1
        if (mod_bw > (2*frame_rate)):
            interp_factor = int(math.ceil(mod_bw/frame_rate)*2)
            init_data = audio_interp_hold(init_data, interp_factor)
            rolloff = int((mod_bw*1.05))
            cutoff = int((mod_bw*1.30))
            init_data = audio_filter(init_data, rolloff, cutoff, frame_rate*interp_factor)
        # modulate the signal
        mod_data = audio_fm_mod(init_data, mod_bw, frame_rate*interp_factor) * scale
        # interpolate the data up to the desired sample rate
        # find the N value that gets us closest to the sample rate
        # this is a shortcut that avoids a lengthy resample process, and as long as fs >> fw
        # the audio impact is small
        N = int(round(samp_rate/(frame_rate*interp_factor)))
        samp_rate_t = (frame_rate*interp_factor)*N
        interp_data = audio_interp_zeros(mod_data, N)
        # IIR filter using SOS method
        # fast, and stable (enough)
        rolloff = int((mod_bw*1.05))
        cutoff = int((mod_bw*1.35))
        filt_data = audio_filter(interp_data, rolloff, cutoff, samp_rate_t)
        # return the filtered data
        return 0, filt_data
    except:
        filt_data = np.zeros(0,dtype="complex64")
        return 1, filt_data

def am_analog_mod(filename, mod_th, samp_rate, scale):
    try:
        # step 1: open the file
        init_data = wav_file_load(filename)
        # modulate the signal
        mod_data = audio_am_mod(init_data, mod_th, frame_rate) * scale
        # interpolate the data up to the desired sample rate
        # find the N value that gets us closest to the sample rate
        # this is a shortcut that avoids a lengthy resample process, and as long as fs >> fw
        # the audio impact is small
        N = int(round(samp_rate/(frame_rate)))
        samp_rate_t = (frame_rate)*N
        interp_data = audio_interp_zeros(mod_data, N)
        # IIR filter using SOS method
        # fast, and stable (enough)
        rolloff = int((5000))
        cutoff = int((10000))
        filt_data = audio_filter(interp_data, rolloff, cutoff, samp_rate_t)
        # return the filtered data
        return 0, filt_data
    except:
        filt_data = np.zeros(0,dtype="complex64")
        return 1, filt_data


def main():
    return 0

if __name__ == '__main__':
    # exit
    sys.exit(main())
