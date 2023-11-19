import sys
import os
from pathlib import Path 
import numpy as np
import multimodal_whisper
sys.path.append(str(Path(__file__).resolve().absolute()))
from whisper_finetune.dataset import valid_audio_text_safe, save_data_list
from whisper_finetune.util import load_wav
import random
#修改这个代码，可以将新获得的一批wav和lrc，加入mel存入lrc和.txt[filename,melpath,lrctxt]文件中;最后删除wav和lrc


def preprocess(wav_path, asr_text, data_dir, audio_max_length, text_max_length):#输入wavpath和相应文本，生成mel保存到data_dir

    out_mel_path = os.path.join(data_dir, "mel/" + os.path.basename(wav_path).replace(".wav",".npy"))
    if not os.path.exists(out_mel_path):#已经生成的mel不再重复生成
        wav = load_wav(wav_path)
        #不符合要求的不保存mel
        if not valid_audio_text_safe(asr_text, wav_path, text_max_length, audio_max_length):
            return None
        
        # wav -> mel
        wav = multimodal_whisper.pad_or_trim(wav.flatten())#只截取了30s?shape:torch.Size([2, 571200])->torch.Size([480000])
        mel = multimodal_whisper.log_mel_spectrogram(wav).numpy().astype(np.float32)#(80, 3000)
        #mkdir mel
        if not os.path.exists(os.path.join(data_dir, "mel")):
            os.mkdir(os.path.join(data_dir, "mel"))

        np.save(out_mel_path, mel)
        return str(out_mel_path)
    else:
        return None#如果已经生成不再返回mel路径
    


def process_dir(data_dir, wav_dir,lrc_dir, audio_max_length=48000, text_max_length=120):
    val_list = []
    train_list=[]
    files = os.listdir(wav_dir)
    files.sort()
    wavs = []#wav文件名
    lrcs=[]#lrc
    for wav_file in files:#过滤不是wav的文件，读取wav对应的lrc
        if wav_file.endswith('.wav'):
            wavs.append(wav_file)
            lrc_path=os.path.join(lrc_dir,wav_file.replace(".wav",".txt"))#相应的lrc路径为
            f=open(lrc_path)
            lrc=f.read()
            lrcs.append(lrc)
    #wavs=["aaa.wav","bbb.wav",...];data=[str1,str2,...]
    assert len(lrcs) == len(wavs)

    for wav_file, asr_text in zip(wavs, lrcs):#先生成新一批mel
        wav_path = os.path.join(wav_dir, wav_file)
        mel_path = preprocess(wav_path, asr_text, data_dir, audio_max_length, text_max_length)
        i=random.random()
        if mel_path!=None:
            if i<0.95:
                train_list.append([wav_file.replace(".wav",""), mel_path, asr_text])#[filename,mel_path,lrctxt]
            else:
                val_list.append([wav_file.replace(".wav",""), mel_path, asr_text])
    return train_list,val_list
# def lrc():
#         #再把根据mel生成data_list->train.txt和val.txt
#     mels=os.listdir(os.path.join(data_dir,"mel"))
#     random.shuffle(mels)
#     train_list=[]
#     val_list=[]
#     for i,mel in enumerate(mels):
#         mel_path=os.path.join(data_dir, "mel/" + mel)
#         lrc_path=os.path.join(lrc_dir,mel.replace(".npy",".txt"))


    
#     return train_list,val_list

def deletedir(dir):
    files=os.listdir(dir)
    for file in files:
        os.remove(os.path.join(dir,file))

if __name__ == '__main__':
    data_dir = "data/processed_1117/"#保存mel和.txt
    # lrc_dir="data/1117/lrc"
    # wav_dir="data/1117/wav"#生成mel之后才对mel分训练集和测试集
    # lrc_dir="cutlrc"
    # wav_dir="cutwav"#生成mel之后才对mel分训练集和测试集
    lrc_dir="cutlrc"
    wav_dir="cutwav"#生成mel之后才对mel分训练集和测试集
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    audio_max_length = 480000
    text_max_length = 120

    #生成mel
    train_list,val_list =process_dir(data_dir,wav_dir,lrc_dir, audio_max_length, text_max_length)

    #生成data_list
    # lrc()
    
    save_data_list(train_list, os.path.join(data_dir, "train.txt"))
    save_data_list(val_list, os.path.join(data_dir, "val.txt"))
    save_data_list(val_list, os.path.join(data_dir, "test.txt"))

    deletedir(wav_dir)
