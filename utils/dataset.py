import math, random

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchaudio
import soundfile as sf
class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['audio_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.fprint_path = data_pd['fprint_path'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def pad_trunc(self,sig, sr, max_ms):

        # sig = np.expand_dims(sig, axis=0)
        # print(sig.shape)
        num_rows, sig_len, = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return sig, sr

    def rechannel(self,sig, sr, new_channel):

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return sig, sr

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return resig, sr

    def resample( self,sig, sr, newsr):

        if (sr == newsr):
            # Nothing to do
            return sig, sr

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    def __getitem__(self, item):
        if self.train:
            audio_path = self.photo_path[item]
            label = self.photo_label[item]
            fprint_path=self.fprint_path[item]
            waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
            gan_waveform=np.load(fprint_path)
            gan_waveform=torch.tensor(gan_waveform)
            waveform, sample_rate  = self.rechannel( waveform, sample_rate ,2)
            waveform, sample_rate = self.resample(waveform, sample_rate ,24000)
            waveform, sample_rate=self.pad_trunc(waveform, sample_rate,2000)

            gan_waveform, sample_rate  = self.rechannel( gan_waveform, sample_rate ,2)
            gan_waveform, sample_rate = self.resample(gan_waveform, sample_rate ,24000)
            gan_waveform, sample_rate=self.pad_trunc(gan_waveform, sample_rate,2000)
            return  waveform, label,gan_waveform
        else:
            audio_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            fprint_path = self.fprint_path[item]
            waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
            gan_waveform = np.load(fprint_path)
            gan_waveform = torch.tensor(gan_waveform)
            # waveform=torch.tensor(waveform)
            # sample_rate=torch.tensor(sample_rate)
            waveform, sample_rate  = self.rechannel(  waveform, sample_rate ,2)
            waveform, sample_rate = self.resample(waveform, sample_rate, 24000)
            waveform, sample_rate=self.pad_trunc(waveform, sample_rate,2000)
            gan_waveform, sample_rate  = self.rechannel( gan_waveform, sample_rate ,2)
            gan_waveform, sample_rate = self.resample(gan_waveform, sample_rate ,24000)
            gan_waveform, sample_rate=self.pad_trunc(gan_waveform, sample_rate,2000)
            # print(waveform.shape)
            # data=torch.tensor(data)
            # data = self.transforms(data)
            return  waveform, label,videoID,gan_waveform
