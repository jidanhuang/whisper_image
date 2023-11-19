import torch
import torchaudio
import tqdm

def load_wav(wav_path, sample_rate:int=16000) -> torch.Tensor:
    """load audio file"""
    waveform, sr = torchaudio.load(wav_path, normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform


def load_txt(txt_path) -> str:
    txt = open(txt_path, "r").read()
    txt = txt.replace("\n", "")

    return txt
import io
import librosa
import torch
import torchaudio
import librosa
import torch
import torchaudio
import os
import numpy as np
import multimodal_whisper

def load_and_convert_to_mel_use_librosa(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
    # Load audio from ogg file
    audio, sr = librosa.load(ogg_path, sr=None, mono=True, duration=duration)#(696960,)
    # Resample if necessary
    if sample_rate != sr:
        audio = librosa.resample(audio, sr, sample_rate)#(232320,)
    # Pad or trim the audio to be exactly the desired duration
    if len(audio) < sample_rate * duration:
        shortage = sample_rate * duration - len(audio)
        audio = np.pad(audio, (0, shortage), mode='constant')
    elif len(audio) > sample_rate * duration:
        audio = audio[:sample_rate * duration]
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mel_spec = multimodal_whisper.log_mel_spectrogram(audio).numpy().astype(np.float32)#(80, 3000)
    mel_spec = torch.from_numpy(mel_spec.astype(np.float32))

    # Convert to tensor and add batch dimension
    # window = torch.hann_window(400).to(audio.device)

    # # Convert to mel spectrogram
    # mel_transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    # mel_spec = mel_transform(audio_tensor)
    # mel_spec=mel_spec[:,:,:3000]
    return mel_spec

    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    mel_spec = mel_transform(audio_tensor)
    mel_spec=mel_spec[:,:,:3000]
    return mel_spec

    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    mel_spec = mel_transform(audio_tensor)
    mel_spec=mel_spec[:,:,:3000]
    return mel_spec


def load_and_convert_to_mel(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
    # Load audio from ogg file
    wav = load_wav(ogg_path)
    # Pad or trim the audio to be exactly the desired duration
    wav = multimodal_whisper.pad_or_trim(wav.flatten())#只截取了30s?shape:torch.Size([2, 571200])->torch.Size([480000])
    mel = multimodal_whisper.log_mel_spectrogram(wav)#(80, 3000)

    return mel

def split_dataset(setname):
    with open('data_narrative/data_list/'+setname+'.txt', 'r') as f:
        image_ids = f.readlines()
        #移除行末换行符
    data_list = [line.rstrip('\n') for line in image_ids]
    # decide the ratio of validation and training data
    val_ratio = 0.1 # for example, 20% for validation and 80% for training

    # get the number of validation and training data
    val_num = int(len(data_list) * val_ratio)
    train_num = len(data_list) - val_num

    # split the list accordingly
    val_list = data_list[:val_num]
    train_list = data_list[val_num:]
    
    return train_list,val_list    

