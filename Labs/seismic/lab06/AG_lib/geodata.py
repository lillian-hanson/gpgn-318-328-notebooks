# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import librosa
from IPython.display import Audio
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt
import soundfile as sf
import warnings

class geodata:
    def __init__(self):
        self.rx = None
        self.rz = None
        self.name = None
        self.delta_t = None
        self.data = None

    def print(self):
        print("----- Information -----")
        print("Name:", self.name)
        print("Data Preview:", self.data[:5])
        

    def distance_to(self, other):
        dx = self.rx - other.rx
        dz = self.rz - other.rz
        return np.sqrt(dx**2 + dz**2)
    
    def plot(self, other=None):
        t = np.arange(len(self.data)) * self.delta_t

        plt.plot(t, self.data, label=self.name)

        if other is not None:
            t2 = np.arange(len(other.data)) * other.delta_t
            plt.plot(t2, other.data, label=other.name)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    def load_w4a(self, filename):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            audio, sampling_rate = librosa.load(filename, sr=None)
        self.data = audio
        self.delta_t = 1.0 / sampling_rate

    def play_audio(self):
        sampling_rate = int(1.0/self.delta_t)
        return Audio(self.data, rate=sampling_rate)
    
    def plot_spectrum(self, logscale=False):
        if self.data is None or self.delta_t is None:
            print("No data to analyze.")
            return

        n = len(self.data)
        freqs = np.fft.rfftfreq(n, d=self.delta_t)
        spectrum = np.abs(np.fft.rfft(self.data))

        plt.figure()
        plt.plot(freqs, spectrum)

        if logscale:
            plt.yscale("log")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum")
        plt.show()
        
    def plot_spectrogram(self, **kwargs):
        if self.data is None or self.delta_t is None:
            print("No data to analyze.")
            return

        fs = 1.0 / self.delta_t

        f, t, Sxx = spectrogram(
            self.data,
            fs=fs,
            **kwargs
        )

        plt.figure()
        plt.pcolormesh(t, f, Sxx, shading="auto")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram")

    def lpfilter(self, fcut, order=4):
        if self.data is None or self.delta_t is None:
            print("No data to filter.")
            return

        fs = 1.0 / self.delta_t          # sampling frequency
        nyq = 0.5 * fs                   # Nyquist frequency

        if fcut >= nyq:
            raise ValueError("Cutoff frequency must be less than Nyquist frequency")

        # Normalized cutoff
        Wn = fcut / nyq

        # Design filter
        b, a = butter(order, Wn, btype="low")

        # Apply filter (zero phase)
        self.data = filtfilt(b, a, self.data)

    @staticmethod
    def convert_phyphox_acceleration(df, time_column=None):
        data_lst = []

        if time_column is not None:
            time = df[time_column].values
            delta_t = np.mean(np.diff(time))
            data_columns = [col for col in df.columns if col != time_column]
        else:
            delta_t = 1.0
            data_columns = df.columns

        for col in data_columns:
            g = geodata()
            g.name = col
            g.data = df[col].values
            g.delta_t = delta_t
            data_lst.append(g)

        return data_lst
import numpy as np
import matplotlib.pyplot as plt

class SeismicGather:
    def __init__(self):
        self.rx = None
        self.rz = None
        self.sx = None
        self.data = None
        self.delta_t = None
        self.offset = None
        self.src_wavelet = None

    def read_synthetic_data(self, filename):
        npz_data = np.load(filename)
        data_length = npz_data['data'][1]
        self.data = []
        for i in range(len(data_length)):
            station = geodata()
            station.rx = npz_data['rx'][i] #x coords
            station.rz = npz_data['rz'][i] # z coords
            station.sx = npz_data['sx'][i] #source x coords
            station.sz = npz_data['sz'][i] # source z coords
            station.offset = np.sqrt((station.rx - station.sx)**2 + (station.rz - station.sz)**2)
            station.delta_t = npz_data['dt'][i]
            station.data = npz_data['data'][:,i]
            if 'src_wavelet' in npz_data.keys():
                station.src_wavelet = npz_data['src_wavelet']
            self.data.append(station)

    def wiggle_plot(self, xaxis='rx', scale=1):
        for data_instance in self.data:
            offset_amt = getattr(data_instance, xaxis)
            data_instance.plot_vertical_trace(scale=scale, offset=offset_amt)
        plt.gca().invert_yaxis()

    def remove_source_wavelet(self):
        for data_instance in self.data:
            xcor = correlate(data_instance.data, data_instance.src_wavelet, 'full')
            xcor = xcor[xcor.size//2:]
            data_instance.data = xcor

    def bpfilter(self, low_freq=False, high_freq=False):
        for data_instance in self.data:
            if high_freq:
                data_instance.lpfilter(cutoff=high_freq)
            if low_freq:
                data_instance.hpfilter(cutoff=low_freq)






                             