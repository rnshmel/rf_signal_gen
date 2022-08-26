# RF Signal Generator

python3-based RF signal generation toolset for R&D

### Version Control Information

Program version: 0.1.1

### TODO List and Known Issues

using the sig_gen tool, users can only input files as signal data - add support for raw hex input

need to add support for pre-ambels/post-ambels/checksums

audio AM/FM files must be WAV files MONO channel, STEREO files crash the tool

audio AM/FM has no baseband center freq select

advanced waveforms need to be written

### Dependencies

sudo apt-get install python3-numpy python3-scipy

### Usage

See user_guide.pdf

### Program Overview

signal_gen.py: user interface for signal generator

mod_utils.py: libray of basic modulation functions

audio_utils.py: library of audio functions

spread_spec\_utils.py: library of spreading functions
