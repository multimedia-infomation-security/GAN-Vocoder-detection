import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset
from utils.utils import audios,audios_ljspeech

def get_dataset(src1_data, src2_data,  src3_data,
                tgt_data, batch_size):
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = audios(flag=0, dataset_name=src1_data)
    src1_train_data_real = audios(flag=1,  dataset_name=src1_data)
    print('Source Data: ', src2_data)
    src2_train_data_fake = audios(flag=0,  dataset_name=src2_data)
    src2_train_data_real = audios(flag=1,  dataset_name=src2_data)
    print('Source Data: ', src3_data)
    src3_train_data_fake = audios(flag=0,  dataset_name=src3_data)
    src3_train_data_real = audios(flag=1,  dataset_name=src3_data)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = audios(flag=2,  dataset_name=tgt_data)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=True)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader


def get_dataset_ljspeech(data,batch_size):
    print('Load Source Data')
    print('Source Data: ', data)
    train_data= audios_ljspeech("train", dataset_name=data)
    val_data=audios_ljspeech("val",dataset_name=data)
    test_data = audios_ljspeech("test",  dataset_name=data)

    train_dataloader = DataLoader(YunpeiDataset(train_data, train=True),
                                            batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(YunpeiDataset(val_data, train=False),
                                            batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader







