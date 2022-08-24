import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import audio_utils as au

"""
gaussian window generation function
returns a standard normal gaussian function curve (PSD)
u = 0, omega^2 = 1, x = [-3,3]
inputs:
window_length - int (odd pref) - length of gaussian window
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
gauss_window - numpy array, float32 - gaussian PSD of window_length
"""
def gauss_window_gen(window_len):
    # gaussian function generator
    # window length is rec. to be odd
    x_axis = np.zeros(window_len,dtype="float32")
    gauss_window = np.zeros(window_len,dtype="float32")
    # step size = [-3,3] / (window_len +1)
    step_size = 6.0/(window_len+1)
    for i in range(0,window_len):
        x_axis[i] = -3.0 + (step_size * (i + 1))
    # function is:
    # 1/omega*sqrt(2pi)  *  exp(-(x - u)^2/2*omega^2)
    # simplified: 1/sqrt(2pi) * exp (-(x^2)/2)
    for i in range(0,window_len):
        gauss_window[i] = (1/math.sqrt(2*math.pi))*math.exp(-(math.pow(x_axis[i],2))/2)
    gauss_window = gauss_window.astype("float32")
    return gauss_window

"""
generate a single complex tone
inputs:
freq - frequency (int)
N - length (int)
samp_rate - sample rate (int)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output np array
"""
def tone_gen(freq, N, samp_rate):
    # function wrapped in try-except
    try:
        # generate a numpy array of length N, filled with freq val
        freq_array = np.full(N, freq, dtype="float32")
        freq_array = freq_array * 2 * np.pi
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(freq_array/samp_rate)
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
generate a single complex chirp
inputs:
start_freq - starting frequency (int)
stop_freq - stoping frequency (int)
N - length in samples (int)
samp_rate - sample rate (int)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output np array
"""
def chirp_gen(start_freq, stop_freq, N, samp_rate):
    # function wrapped in try-except
    try:
        # generate a numpy array of length N, filled with freq val
        step = float((stop_freq - start_freq)/N)
        print(step)
        freq_array = np.arange(start_freq, stop_freq, step, dtype="float32")
        freq_array = freq_array * 2 * np.pi
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(freq_array/samp_rate)
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
basic 2fsk modulation: non-coherent
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
freq_div - int - frequency spacing between low and high freq (BW)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def fsk_mod_2(raw_data, samp_rate, baud_rate, freq_div, center_freq):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        expanded_array = np.zeros(len(raw_data)*8*sps,dtype = "int32") # bit-expanded array
        pos = 0 # variable to store position in expanded array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8):
                # for each bit, shift left and AND with 128
                # if value is 0, shifted bit was zero
                # else, shifted bit was 1
                bit = raw_data[i] << j & 128
                if bit > 0:
                    # if > 0: add samp_rate number of 1's to expanded array
                    # else: add samp_rate number of 0's to expanded array
                    for k in range(0,sps):
                        expanded_array[pos+k] = 1
                    pos = pos + sps
                else:
                    for k in range(0,sps):
                        expanded_array[pos+k] = -1
                    pos = pos + sps
        # multiply expanded array by freq_div/2
        expanded_array = (expanded_array * freq_div/2)+center_freq
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(expanded_array*2*np.pi)/samp_rate
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
basic 4fsk modulation: non-coherent
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
freq_div - int - frequency spacing between low and high freq (BW)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def fsk_mod_4(raw_data, samp_rate, baud_rate, freq_div, center_freq):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        expanded_array = np.zeros(len(raw_data)*4*sps,dtype = "float32") # bit-expanded array
        pos = 0 # variable to store position in expanded array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8,2):
                # for each bit, shift left and AND with 192
                # 4 cases:
                # 0 = 00
                # 64 = 01
                # 128 = 10
                # 192 = 11
                bit = raw_data[i] << j & 192
                if bit == 0:
                    # case 00 (-1)
                    for k in range(0,sps):
                        expanded_array[pos+k] = -1.0
                    pos = pos + sps
                elif bit == 64:
                    # case 01 (-.33)
                    for k in range(0,sps):
                        expanded_array[pos+k] = -0.33
                    pos = pos + sps
                elif bit == 128:
                    # case 10 (.33)
                    for k in range(0,sps):
                        expanded_array[pos+k] = 0.33
                    pos = pos + sps
                else:
                    # case 11 (1)
                    for k in range(0,sps):
                        expanded_array[pos+k] = 1.0
                    pos = pos + sps
        # multiply expanded array by freq_div/2
        expanded_array = (expanded_array * freq_div/2)+center_freq
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(expanded_array*2*np.pi)/samp_rate
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
gaussian 2fsk modulation
2FSK with a guassian window function applied
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
freq_div - int - frequency spacing between low and high freq (BW)
window_len = int - gaussian function window length in samples
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def gfsk_mod_2(raw_data, samp_rate, baud_rate, freq_div, center_freq, window_len):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        expanded_array = np.zeros(len(raw_data)*8*sps,dtype = "int32") # bit-expanded array
        pos = 0 # variable to store position in expanded array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8):
                # for each bit, shift left and AND with 128
                # if value is 0, shifted bit was zero
                # else, shifted bit was 1
                bit = raw_data[i] << j & 128
                if bit > 0:
                    # if > 0: add samp_rate number of 1's to expanded array
                    # else: add samp_rate number of 0's to expanded array
                    for k in range(0,sps):
                        expanded_array[pos+k] = 1
                    pos = pos + sps
                else:
                    for k in range(0,sps):
                        expanded_array[pos+k] = -1
                    pos = pos + sps
        # generate gaussian window
        gauss_window = gauss_window_gen(window_len)
        # apply gaussian window to signal
        expanded_array = np.convolve(expanded_array,gauss_window, mode='same')
        # normalize array (-1 to 1)
        expanded_array = expanded_array/np.amax(expanded_array)
        # multiply expanded array by freq_div/2 + center freq
        expanded_array = (expanded_array * freq_div/2)+center_freq
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(expanded_array*2*np.pi)/samp_rate
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
gaussian 4fsk modulation
2FSK with a guassian window function applied
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
freq_div - int - frequency spacing between low and high freq (BW)
window_len = int - gaussian function window length in samples
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def gfsk_mod_4(raw_data, samp_rate, baud_rate, freq_div, center_freq, window_len,):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        expanded_array = np.zeros(len(raw_data)*4*sps,dtype = "float32") # bit-expanded array
        pos = 0 # variable to store position in expanded array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8,2):
                # for each bit, shift left and AND with 192
                # 4 cases:
                # 0 = 00
                # 64 = 01
                # 128 = 10
                # 192 = 11
                bit = raw_data[i] << j & 192
                if bit == 0:
                    # case 00 (-1)
                    for k in range(0,sps):
                        expanded_array[pos+k] = -1.0
                    pos = pos + sps
                elif bit == 64:
                    # case 01 (-.33)
                    for k in range(0,sps):
                        expanded_array[pos+k] = -0.33
                    pos = pos + sps
                elif bit == 128:
                    # case 10 (.33)
                    for k in range(0,sps):
                        expanded_array[pos+k] = 0.33
                    pos = pos + sps
                else:
                    # case 11 (1)
                    for k in range(0,sps):
                        expanded_array[pos+k] = 1.0
                    pos = pos + sps
        # generate gaussian window
        gauss_window = gauss_window_gen(window_len)
        # apply gaussian window to signal
        expanded_array = np.convolve(expanded_array,gauss_window, mode='same')
        # normalize array (-1 to 1)
        expanded_array = expanded_array/np.amax(expanded_array)
        # multiply expanded array by freq_div/2
        expanded_array = (expanded_array * freq_div/2)+center_freq
        # integrate frequency to calculate the phase at any given sample value
        phi_array = np.cumsum(expanded_array*2*np.pi)/samp_rate
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
basic bpsk modulation: non-coherent
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def bpsk_mod(raw_data, samp_rate, baud_rate, center_freq):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        phi_array = np.zeros(len(raw_data)*8*sps,dtype = "float32") # bit-expanded array
        pos = 0 # variable to store position in phi array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8):
                # for each bit, shift left and AND with 128
                # if value is 0, shifted bit was zero
                # else, shifted bit was 1
                bit = raw_data[i] << j & 128
                if bit > 0:
                    # if > 0: add samp_rate number of 1's to phi array
                    # else: add samp_rate number of 0's to phi array
                    for k in range(0,sps):
                        phi_array[pos+k] = np.pi
                    pos = pos + sps
                else:
                    for k in range(0,sps):
                        phi_array[pos+k] = 0.0
                    pos = pos + sps
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # check if we need to RF mix off baseband
        if (center_freq != 0):
            exit, mix_sig = tone_gen(center_freq, len(complex_array), samp_rate)
            if (exit == 0):
                # success, mix
                complex_array = complex_array * mix_sig
            else:
                # error with tone
                # return 1 and an empty array
                complex_array = np.zeros(0,dtype="complex64")
                return 1, complex_array
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array


"""
basic qpsk modulation: non-coherent
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def qpsk_mod(raw_data, samp_rate, baud_rate, center_freq):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        phi_array = np.zeros(len(raw_data)*4*sps,dtype = "float32") # bit-expanded array
        pos = 0 # variable to store position in expanded array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8,2):
                # for each bit, shift left and AND with 192
                # 4 cases:
                # 0 = 00
                # 64 = 01
                # 128 = 10
                # 192 = 11
                bit = raw_data[i] << j & 192
                if bit == 0:
                    # case 00 (-1)
                    for k in range(0,sps):
                        phi_array[pos+k] = np.pi/4.0
                    pos = pos + sps
                elif bit == 64:
                    # case 01 (-.33)
                    for k in range(0,sps):
                        phi_array[pos+k] = 3.0*np.pi/4.0
                    pos = pos + sps
                elif bit == 128:
                    # case 10 (.33)
                    for k in range(0,sps):
                        phi_array[pos+k] = 5.0*np.pi/4.0
                    pos = pos + sps
                else:
                    # case 11 (1)
                    for k in range(0,sps):
                        phi_array[pos+k] = 7.0*np.pi/4.0
                    pos = pos + sps
        # get i values with cosine
        cos_array = np.cos(phi_array)
        # get j values with sine
        sin_array = np.sin(phi_array)
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (cos_array + 1j*sin_array).astype("complex64")
        # check if we need to RF mix off baseband
        if (center_freq != 0):
            exit, mix_sig = tone_gen(center_freq, len(complex_array), samp_rate)
            if (exit == 0):
                # success, mix
                complex_array = complex_array * mix_sig
            else:
                # error with tone
                # return 1 and an empty array
                complex_array = np.zeros(0,dtype="complex64")
                return 1, complex_array
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
basic ook modulation
inputs:
raw_data - bytearray - the raw binary data to modulate
samp_rate - int - sample rate of end IQ data
baud_rate - int - baud rate of input data (samples per symbol)
outputs:
exit_code - int - function return code: 0 for success, 1+ for errors
iq_data - numpy array, complex64 - output modulation data
"""
def ook_mod(raw_data, samp_rate, baud_rate, center_freq):
    # function wrapped in try-except
    try:
        sps = int(samp_rate/baud_rate) # samples per symbol
        amp_array = np.zeros(len(raw_data)*8*sps,dtype = "float32") # bit-expanded array
        pos = 0 # variable to store position in phi array
        # loop through the bytearray raw data
        for i in range(0, len(raw_data)):
            # loop through each bit
            for j in range(0,8):
                # for each bit, shift left and AND with 128
                # if value is 0, shifted bit was zero
                # else, shifted bit was 1
                bit = raw_data[i] << j & 128
                if bit > 0:
                    # if > 0: add samp_rate number of 1's to phi array
                    # else: add samp_rate number of 0's to phi array
                    for k in range(0,sps):
                        amp_array[pos+k] = 1.0
                    pos = pos + sps
                else:
                    for k in range(0,sps):
                        amp_array[pos+k] = 0.0
                    pos = pos + sps
        # generate a tone
        if (center_freq != 0):
            exit, tone_sig = tone_gen(center_freq, len(amp_array), samp_rate)
            if (exit != 0):
                # error with tone
                # return 1 and an empty array
                complex_array = np.zeros(0,dtype="complex64")
                return 1, complex_array
        else:
            exit, tone_sig = tone_gen(0, len(amp_array), samp_rate)
            if (exit != 0):
                # error with tone
                # return 1 and an empty array
                complex_array = np.zeros(0,dtype="complex64")
                return 1, complex_array
        # multiply amp array with tone array
        # combine into a complex array
        # save as complex64 (default is complex128)
        complex_array = (tone_sig * amp_array).astype("complex64")
        # return exit code 0 and complex array
        return 0, complex_array
    except:
        # error handle
        # return 1 and an empty array
        complex_array = np.zeros(0,dtype="complex64")
        return 1, complex_array

"""
main function for testing purpose
"""
def main():
    return 0

if __name__ == '__main__':
    # define inputs
    raw_data = bytearray("UUUU#this is a test message 01234567890123456789#","ASCII")
    #raw_data = bytearray([0,0,228,228,228,228,255,255,0,0,228,228,228,228,255,255,0,0,228,228,228,228,255,255])
    #raw_data = bytearray([255,255,0,0,170,170,255,255,0,0,170,170,255,255,0,0,170,170,255,255,0,0,170,170])
    """
    raw_data = bytearray(255)
    for i in range(0,len(raw_data)):
        raw_data[i] = i
    """
    samp_rate = 250000
    baud_rate = 300
    freq_div = 100000
    center_freq = 0
    sps = int(samp_rate/baud_rate)

    window_len = int(.15*sps)
    if (window_len % 2) == 0:
        window_len = window_len + 1
    print("generator signal values")
    print(samp_rate)
    print(baud_rate)
    print(freq_div)
    print(center_freq)
    print(sps)
    print(window_len)

    # call
    #exit, data = bpsk_mod(raw_data, samp_rate, baud_rate, center_freq)
    #exit, data = bpsk_mod_lsl(raw_data, samp_rate, baud_rate, center_freq, 31)
    #exit, data = tone_gen(50000, samp_rate, samp_rate)
    #exit, data = gfsk_mod_4(raw_data, samp_rate, baud_rate, freq_div, center_freq, window_len)
    exit, data = chirp_gen(25000, -25000, 1000000, 200000)

    #filename = "/home/user/Downloads/flock_of_seagulls.wav"
    #filename = "/home/user/Downloads/pcm0808m.wav"
    #filename = "/home/user/Downloads/pcm1608m.wav"
    #filename = "test.wav"
    """
    print(filename)
    audio_data = au.wav_file_load(filename)
    proc_data = au.wav_file_upsample(audio_data, samp_rate)
    exit, data = fm_mod(audio_data, samp_rate, freq_div, center_freq)
    """

    if exit == 0:
        print("success, writing to file - len: "+str(len(data)))
        data.tofile("test.iq")
        temp = data[1:] * np.conj(data[:-1])
        #demod = np.abs(np.angle(temp))
        demod = np.angle(temp)
        plt.plot(demod)
        plt.show()
    else:
        print("error in test function, not writing to file")
    
    # exit
    sys.exit(main())