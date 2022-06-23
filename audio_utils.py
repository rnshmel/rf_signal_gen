import numpy as np
import matplotlib.pyplot as plt
import wave

# globals
ch_num = 0
samp_width = 0
frame_num = 0
frame_rate = 0

def wav_file_resample(y_data, I_val, D_val):
    interp_data = np.zeros(len(y_data)*I_val, dtype="float32")
    for i in range(0, len(y_data)):
        for j in range(0, I_val):
            interp_data[(i*I_val)+j] = y_data[i]
    dec_data = interp_data[0:len(interp_data):D_val]
    return dec_data

def wav_file_upsample(y_data, sample_rate):
    I_val = int(round(sample_rate/frame_rate))
    interp_data = np.zeros(len(y_data)*I_val, dtype="float32")
    for i in range(0, len(y_data)):
        for j in range(0, I_val):
            interp_data[(i*I_val)+j] = y_data[i]
    return interp_data

# loads a wav file
# file can be 8 or 16 bit audio - MONO
# returns the file in a np array (dtype = float32)
def wav_file_load(filename):
    global ch_num
    global samp_width
    global frame_num
    global frame_rate
    # open wav file - read
    wav_obj_r = wave.open(filename, mode='rb')
    # get basic params
    ch_num = wav_obj_r.getnchannels()
    samp_width = wav_obj_r.getsampwidth()
    frame_num = wav_obj_r.getnframes()
    frame_rate = wav_obj_r.getframerate()

    # read byte data into bytes object
    wav_data_raw = wav_obj_r.readframes(frame_num)
    # depending on sample width, the data is either
    # 1) uint 8 
    # 2) signed int 16
    # convert to float
    if (samp_width == 1):
        # uint8
        wav_data_1 = np.zeros(frame_num, dtype="uint8")
        for i in range(0, len(wav_data_raw)):
            wav_data_1[i] = wav_data_raw[i]
        wav_data_float = wav_data_1.astype("float32") - 127
        wav_data_float = wav_data_float / 255
        wav_data_float = wav_data_float / np.max(wav_data_float)
    else:
        # int16
        wav_data_2 = np.zeros(frame_num, dtype="int16")
        count = 0
        for i in range(0, len(wav_data_raw), 2):
            sample = (wav_data_raw[i+1] << 8) | wav_data_raw[i]
            wav_data_2[count] = sample
            count = count + 1
        wav_data_float = wav_data_2.astype("float32")/np.max(wav_data_2)
    return wav_data_float

"""
def main():
    #filename = "/home/user/Downloads/flock_of_seagulls.wav"
    #filename = "/home/user/Downloads/pcm0808m.wav"
    #filename = "/home/user/Downloads/pcm1608m.wav"
    filename = "test.wav"
    data = wav_file_load(filename)
    print(filename)
    print("ch_num: " + str(ch_num))
    print("samp_width: " + str(samp_width))
    print("frame_num: " + str(frame_num))
    print("frame_rate: " + str(frame_rate))
    proc_data = wav_file_resample(data, 1000, 32)
    return 0

if __name__ == '__main__':
    # exit
    sys.exit(main())
"""