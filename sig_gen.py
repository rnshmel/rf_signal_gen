# signal generator interface
# made to allow easiers use of sig gen libs
import mod_utils
import audio_utils
import sys

program_version = "0.1.1"

def open_file(filename):
    try:
        print("opening file: "+filename)
        with open(filename, "rb") as f:
            data = bytearray(f.read())
        return 0, data
    except:
        return 1, 0

def sing_tone_sel():
    print("\nsingle tone selected\n")
    freq = int(input("baseband frequency (Hz): "))
    N = int(input("signal length (number-of-samples): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.tone_gen(freq,N,samp_rate)
    return exit, data

def swept_tone_sel():
    print("\nswept tone selected\n")
    str_freq = int(input("start baseband frequency (Hz): "))
    stp_freq = int(input("stop baseband frequency (Hz): "))
    N = int(input("signal length (number-of-samples): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.chirp_gen(str_freq,stp_freq,N,samp_rate)
    return exit, data

def analog_am_sel():
    print("\nanalog AM selected\n")
    filename = str(input("WAV file name: "))
    mod_th = int(input("AM modulation threshold (0 - 1): "))
    samp_rate = int(input("sample rate (sps): "))
    scale = int(input("audio scale factor (int): "))
    exit, data = audio_utils.am_analog_mod(filename, mod_th, samp_rate, scale)
    return exit, data

def analog_fm_sel():
    print("\nanalog FM selected\n")
    filename = str(input("WAV file name: "))
    mod_bw = int(input("FM modulation bandwidth (Hz): "))
    samp_rate = int(input("sample rate (sps): "))
    scale = int(input("audio scale factor (int): "))
    exit, data = audio_utils.fm_analog_mod(filename, mod_bw, samp_rate, scale)
    return exit, data

def ook_sel():
    print("\non-off keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.ook_mod(raw_data,samp_rate,baud_rate,freq)
    return exit, data

def fsk2_sel():
    print("\n2-level frequency keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    div = int(input("waveform frequency bandwidth (Hz): "))
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.fsk_mod_2(raw_data,samp_rate,baud_rate,div,freq)
    return exit, data

def fsk4_sel():
    print("\n4-level frequency keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    div = int(input("waveform frequency bandwidth (Hz): "))
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.fsk_mod_4(raw_data,samp_rate,baud_rate,div,freq)
    return exit, data

def gfsk2_sel():
    print("\n2-level gaussian frequency keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    div = int(input("waveform frequency bandwidth (Hz): "))
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    window_per = float(input("gaussian window percentage (0.0 through 1.0): "))
    sps = int(samp_rate/baud_rate)
    window_len = int(window_per*sps)
    if (window_len % 2) == 0:
        window_len = window_len + 1
    exit, data = mod_utils.gfsk_mod_2(raw_data,samp_rate,baud_rate,div,freq,window_len)
    return exit, data

def gfsk4_sel():
    print("\n4-level gaussian frequency keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    div = int(input("waveform frequency bandwidth (Hz): "))
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    window_per = float(input("gaussian window percentage (0.0 through 1.0): "))
    sps = int(samp_rate/baud_rate)
    window_len = int(window_per*sps)
    if (window_len % 2) == 0:
        window_len = window_len + 1
    exit, data = mod_utils.gfsk_mod_4(raw_data,samp_rate,baud_rate,div,freq,window_len)
    return exit, data

def bpsk_sel():
    print("\nbinary phase keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.bpsk_mod(raw_data,samp_rate,baud_rate,freq)
    return exit, data

def qpsk_sel():
    print("\nquadrature phase keying selected\n")
    filename = input("waveform data (full filename path unless in local directory): ")
    fext, raw_data = open_file(filename)
    if (fext != 0):
        print("\nerror with opening file\n")
        return 1, 0
    freq = int(input("baseband center frequency (Hz): "))
    baud_rate = int(input("signal baud rate (symbols-per-second): "))
    samp_rate = int(input("sample rate (sps): "))
    exit, data = mod_utils.qpsk_mod(raw_data,samp_rate,baud_rate,freq)
    return exit, data

def noise_sel():
    print("\nuniform noise signal selected\n")
    N = int(input("length of noise signal (samples): "))
    thresh = float(input("nouse signal threshold (float)(ex. .001 or.01): "))
    exit, data = mod_utils.noise_gen(N, thresh)
    return exit, data

def guard_sel():
    print("\nblank/guard signal selected\n")
    N = int(input("length of blank guard signal (samples): "))
    exit, data = mod_utils.blank_gen(N)
    return exit, data

def fhss_sel():
    print("\nFHSS selected\n")
    print("PLACEHOLDER - function not written yet")
    exit, data = 1, 0
    return exit, data

def dsss_sel():
    print("\nDSSS selected\n")
    print("PLACEHOLDER - function not written yet")
    exit, data = 1, 0
    return exit, data

def ofdm_sel():
    print("\nOFDM selected\n")
    print("PLACEHOLDER - function not written yet")
    exit, data = 1, 0
    return exit, data

def main():
    print("Signal Generator Tool")
    print("version: "+program_version)
    print("\n===== basic waveforms =====")
    print("0:   single tone")
    print("1:   swept tone")
    print("2:   analog AM (voice)")
    print("3:   analog FM (voice)")
    print("4:   on-off-keyed (OOK)")
    print("5:   2-level Frequency Shift Keyed (2FSK)")
    print("6:   4-level Frequency Shift Keyed (4FSK)")
    print("7:   2-level Gaussian Frequency Shift Keyed (2GFSK)")
    print("8:   4-level Gaussian Frequency Shift Keyed (4GFSK)")
    print("9:   Binary Phase Shift Keyed (BPSK)")
    print("10:  Quadrature Phase Shift Keyed (QPSK)")
    print("11:  Noise Signal (uniform distribution)")
    print("12:  Guard Band (no signal)")
    print("\n===== advanced waveforms =====")
    print("13:  Frequency Hopping Spread Spectrum (FHSS)")
    print("14:  Direct Sequence Spread Spectrum (DSSS)")
    print("15:  Orthogonal Frequency Division Multiplexing (OFDM)")

    user_choice = int(input("\nselect a signal to generate: "))

    if user_choice == 0:
        exit, data = sing_tone_sel()
    elif user_choice == 1:
        exit, data = swept_tone_sel()
    elif user_choice == 2:
        exit, data = analog_am_sel()
    elif user_choice == 3:
        exit, data = analog_fm_sel()
    elif user_choice == 4:
        exit, data = ook_sel()
    elif user_choice == 5:
        exit, data = fsk2_sel()
    elif user_choice == 6:
        exit, data = fsk4_sel()
    elif user_choice == 7:
        exit, data = gfsk2_sel()
    elif user_choice == 8:
        exit, data = gfsk4_sel()
    elif user_choice == 9:
        exit, data = bpsk_sel()
    elif user_choice == 10:
        exit, data = qpsk_sel()
    elif user_choice == 11:
        exit, data = noise_sel()
    elif user_choice == 12:
        exit, data = guard_sel()
    elif user_choice == 13:
        exit, data = fhss_sel()
    elif user_choice == 14:
        exit, data = dsss_sel()
    elif user_choice == 15:
        exit, data = ofdm_sel()
    else:
        print("\ninvalid choice - exiting")
        sys.exit()
    
    if exit != 0:
        print("\nerror in generator function - exiting")
        sys.exit()
    else:
        print("\ngenerator function successful")
        len_b = len(data)*8
        len_kb = len_b/1000.0
        len_mb = len_b / 1000000.0
        print("size in memory: "+str(len_b)+"/"+str(len_kb)+"/"+str(len_mb)+" B/KB/MB")
    
    print("\noutput the complex 64-bit array to a file")
    print("NOTE - file name should include the file type for the use case.")
    print(".fc32, .cf32, and .iq are all common file types for complex data")
    out_name = input("\noutput file name: ")

    print("\nenter output file path, leave blank to save in the local directory")
    out_path = input("\noutput file path: ")

    try:
        data.tofile(out_path+out_name)
        print("file write successful")
    except:
        print("error in file write - exiting")
        sys.exit()
    
    print("\nsignal generation complete - exiting")

if __name__ == "__main__":
    sys.exit(main())
