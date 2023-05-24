# Vocoder Detection  Based on GAN Fingerprints and Domain Generalization

## About

This study proposes vocoder detection of spoofed speech based on GAN fingerprints and domain generalization. The framework consists of three modules:  GAN Fingerprint Extraction and Enhancement,  Domain Generalization and Curriculum Learning, which achieves the most advanced detection results for unknown GAN vocoder detection.

## Model Architecture

![1684895811816](C:\Users\大艺术家\AppData\Roaming\Typora\typora-user-images\1684895811816.png)

## Datasets

Real speech: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [CSMSC](https://www.data-baker.com/open_source.html) , [JUST](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) , [KSS](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)

Forged speech: There are four different GAN vocoders for forged-speech synthesis: [Parallel WaveGAN](https://arxiv.org/abs/1910.11480), [HifiGAN](https://arxiv.org/abs/2010.05646) , [Style-MelGAN](https://arxiv.org/abs/2011.01557) , [MelGAN](https://arxiv.org/abs/1910.06711). 
The pre-training model of each GAN vocoder can be find [here](https://github.com/kan-bayashi/ParallelWaveGAN).
Four datasets were formed: LJSpeech and its MelGAN-synthesized speech , CSMSC and its Style-MelGAN-synthesized speech , JUST and its HifiGAN-synthesized speech, KSS and its Parallel WaveGAN-synthesized speech . 

## Requirement

- python 3.6
- pytorch 1.8.1
- torchaudio 0.8.1
- cuda 10.1

## Usage

### GAN fingerprint generation