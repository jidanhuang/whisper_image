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
# def load_and_convert_to_mel(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
#     # Load audio from ogg file
#     audio, sr = librosa.load(ogg_path, sr=None, mono=True, duration=duration)#(696960,)
#     # Resample if necessary
#     if sample_rate != sr:
#         audio = librosa.resample(audio, sr, sample_rate)#(232320,)
#     # Pad or trim the audio to be exactly the desired duration
#     if len(audio) < sample_rate * duration:
#         shortage = sample_rate * duration - len(audio)
#         audio = np.pad(audio, (0, shortage), mode='constant')
#     elif len(audio) > sample_rate * duration:
#         audio = audio[:sample_rate * duration]

#     # Convert to tensor and add batch dimension
#     audio_tensor = torch.from_numpy(audio).unsqueeze(0)
#     window = torch.hann_window(400).to(audio.device)

#     # Convert to mel spectrogram
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
#     mel_spec = mel_transform(audio_tensor)
#     mel_spec=mel_spec[:,:,:3000]
#     return mel_spec

def load_and_convert_to_mel(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
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

# def load_and_convert_to_mel(ogg_bytes, sample_rate=16000, n_mel_channels=80):
#     # Load audio from memory
#     audio, sr = librosa.load(io.BytesIO(ogg_bytes), sr=None, mono=True)

#     # Resample if necessary
#     if sample_rate != sr:
#         audio = librosa.resample(audio, sr, sample_rate)

#     # Convert to tensor and add batch dimension
#     audio_tensor = torch.from_numpy(audio).unsqueeze(0)

#     # Convert to mel spectrogram
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=sample_rate, n_mels=n_mel_channels)
#     mel_spec = mel_transform(audio_tensor)

#     return mel_spec


