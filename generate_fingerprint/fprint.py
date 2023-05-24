import os

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)
torch.random.manual_seed(0)
from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt
import glob


torch.random.manual_seed(0)

def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)
n_fft = 1024
win_length = None
hop_length = 512


spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)
def syns(audio_path,save_path):
    path_list = glob.glob(audio_path + '*.wav', recursive=True)
    for i in range(len(path_list)):
        print(path_list[i])
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(path_list[i])
        spec=spectrogram(SPEECH_WAVEFORM)
        flag = path_list[i].find('_gen')
        # print(flag)
        if (flag != -1):
            reconstructed_waveform =  T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            length=SPEECH_WAVEFORM.size(1)
            )(spec)
            reconstructed_spec = spectrogram(reconstructed_waveform)
            gan_spec=spec-reconstructed_spec
            np.save(save_path + path_list[i].split('/')[-1].split('.')[0] + ".npy", gan_spec)
        else:
            gan_spec=spec-spec
            np.save(save_path + path_list[i].split('/')[-1].split('.')[0] + ".npy", gan_spec)

if __name__=="__main__":
    path ="/home/lifan/h/audio/"
    save_path="/home/lifan/h/audio/gan_fprint/"
    syns(path,save_path)