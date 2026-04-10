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


    def apply_timeshift(trc,dt):
        taxis = trc.gettaxis()
        trc.data = np.interp(taxis-dt,taxis,trc.data,left=0.right=0)

    def get_taxis(self):
        taxis = np.arange(len(self.data))*self.delta_t
        return taxis


    def plot_interactive(self):
        df = pd.DataFrame()
        df['time'] = self.get_taxis()
        df['v'] = self.data
        fig = px.line(data_frame=df,x='time',y='v')
        fig.show()

    def read_synthetic_data(self, filename):
        data = np.load(filename)
        # Correct mapping discovered during analysis
        # Key: data, Shape: (2410, 20)
        self.data = data['data']
        self.dt = data['dt']
        self.delta_t = data['dt']
        self.x = data['rx']
        self.sx = data['sx']
        
    def plot_interactive(self, trace_index=None):
        if trace_index is not None and self.data.ndim > 1:
            v_data = self.data[:, trace_index]
        else:
            v_data = self.data
        
        df = pd.DataFrame({'time': self.get_taxis(), 'v': v_data})
        fig = px.line(data_frame=df, x='time', y='v')
        fig.show()

    def wiggle_plot(self, scale=1.0):
        if self.data is None:
            print("No data loaded.")
            return
        plt.figure(figsize=(10, 6))
        num_traces = self.data.shape[1] if self.data.ndim > 1 else 1
        t = self.get_taxis()
        for i in range(num_traces):
            trace = self.data[:, i] if num_traces > 1 else self.data
            plt.plot(trace * scale + i, t, 'k', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel('Trace Number')
        plt.ylabel('Time (s)')
        plt.show()

    def lpfilter(self, fcut, order=4): pass
    def play_audio(self): pass
    def plot(self, other=None): pass
    def plot_spectrogram(self, **kwargs): pass
    def plot_spectrum(self, logscale=False): pass
    def distance_to(self, other): pass
    @staticmethod
    def convert_phyphox_acceleration(df, time_column=None): pass

    
        