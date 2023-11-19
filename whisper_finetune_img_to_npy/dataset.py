
from email.mime import audio
import torch
from .frontend import TextFrontend
from .util import load_wav, load_txt,load_and_convert_to_mel
import multimodal_whisper
import numpy as np
from typing import Union
from pathlib import Path
import pickle
import jsonlines
import os 
from pydub import AudioSegment
from PIL import Image
def load_data_img(img_type:str):

    # check 
    # load image_features
    if img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif img_type == "detr":
        image_features = np.load('vision_features/detr.npy')
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", image_features.shape)
    return image_features

class WhisperASRDataset(torch.utils.data.Dataset):
    def __init__(
        self, id_mel_text_list: list,
        tokenizer: multimodal_whisper.tokenizer,
        audio_features_dir:str,
        img_type:str,
        setname: str#sampling_rate: int=16000,
    ) -> None:
        super().__init__()

        # assert len(id_data_list) > 0
        #assert sampling_rate > 0

        # self.id_mel_text_list = id_mel_text_list
        #self.sampling_rate = sampling_rate
        self.tokenizer = tokenizer
        self.audio_features_dir=audio_features_dir
        self.img_type=img_type
        # self.reader= jsonlines.open(os.path.join('data_narrative','jsonl',setname+'.jsonl'))
        self.setname=setname
        # self.image_features=load_data_img(self.img_type)
        # with open('vision_features/detr.pkl', 'rb') as f:#{0:tensorlist,...}
        #     self.image_features = pickle.load(f)
        # with open('data_narrative/image_list.txt', 'r') as f:
        #     image_ids = f.readlines()
        #读取data_list
        with open('data_narrative/data_list/'+setname+'.txt', 'r') as f:
            image_ids = f.readlines()

        #移除行末换行符
        self.data_list = [line.rstrip('\n') for line in image_ids]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, id):#按照jsonl里的顺序加载
        # self.reader.close()  # close the current file pointer
        # self.reader = jsonlines.open(os.path.join('data_narrative', 'jsonl', self.setname+'.jsonl'))
        # self.reader.seek(0)
        # image_id=next(self.reader)['image_id']
        # image_id=self.reader[id]['image_id']
        image_id,text,url=self.data_list[id].split('\t')
        # text
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)#<sot><lang><task><notimestamps><text>
        labels = text[1:] + [self.tokenizer.eot]#<lang><task><notimestamps><text...<eot>

        # mel
        mel=load_and_convert_to_mel(os.path.join('data_narrative/ogg',self.setname,image_id.split('/')[-1]+'.ogg'))
        # ogg_file = AudioSegment.from_ogg(os.path.join('data_narrative/ogg',self.setname,image_id.split('/')[-1]+'.ogg'))
        # wav = ogg_file.set_frame_rate(16000)
        # wav = whisper.pad_or_trim(wav.flatten())#只截取了30s?shape:torch.Size([2, 571200])->torch.Size([480000])
        # mel = whisper.log_mel_spectrogram(wav).numpy().astype(np.float32)#(80, 3000)
        
        # img
        img=Image.open(os.path.join('data_narrative/images',self.setname,image_id.split('/')[-1]+'.jpg'))
        
        # with open('vision_features/detr.pkl', 'rb') as f:#{0:tensorlist,...}
        #     image_features = pickle.load(f)

        # image_id=self.image_features[id,:,:]#(100, 256)
        # image_id=self.image_features[id]#一个样本的image_id=image_tensorlist,tensor_size(100,256)

        # image_features=image_features.reshape(1,image_features.shape[0],image_features.shape[1])#n_array,detr下我们只取第一个一个图片
        #也就是说3个音频的图片是一样的，到后面实操的时候会根据图片npy文件路径来读取（也可以把.npy文件保存为(1,100,256)格式
        return {
            "input_ids": mel,
            "labels": labels,#无sot，有eot
            "dec_input_ids": text,#有sot，无eot
            "img": img#tensorlist,tensor_size(100,256)
        }


class WhisperASRDataCollator():
    def __call__(self, features):#[sample1(dict),sample2,...]feature对应__getitem__返回的batch

        input_ids, labels, dec_input_ids,imgs = [], [], [],[]
        for feature in features:
            input_ids.append(feature["input_ids"])
            labels.append(feature["labels"])
            dec_input_ids.append(feature["dec_input_ids"])
            imgs.append(feature["img"])

        
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        # input_ids = torch.concat(input_ids,dim=0)
        label_lengths = [len(label) for label in labels]
        dec_input_ids_length = [len(dec_input_id) for dec_input_id in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
            for lab, lab_len in zip(labels, label_lengths)
        ]#用-100补齐
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
        }

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False)
            for k, v in batch.items()
        }
        batch["input_ids"] = input_ids        
        batch["imgs"] = imgs        
        # batch["image_features"] = torch.tensor(image_features,requires_grad=False)

        return batch


def valid_audio_text_safe(
        text, audio,
        text_max_length, audio_max_sample_length
    ):    
    if len(text) == 0:
        return False
    if len(text) > text_max_length:
        return False
    if audio is None:
        return False
    if len(audio) > audio_max_sample_length:
        return False
    return True

def save_data_list(
    data_list: list,
    list_path: Union[Path, str]
):
    with open(list_path, "w") as f:
        f.writelines("\t".join(x) + "\n" for x in data_list)


def load_data_list(
    list_path: Union[Path, str]
):
    return [
        x.strip("\n").split("\t")
        for x in open(list_path, "r").readlines()
    ]
