import multimodal_whisper
from pytorch_lightning import LightningModule
import evaluate

import torch
from torch import nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
# import sys
# sys.path.append('data_narrative/detr/')

# import process_image  
from .dataset import WhisperASRDataset, WhisperASRDataCollator
from .util import split_dataset,summary
import csv
import os
import urllib.request
import copy
from dataclasses import dataclass
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


testdim=ModelDimensions(n_audio_ctx=1500,n_audio_head=8,n_audio_layer=6,n_audio_state=512,n_mels=80,n_text_ctx=448,n_text_head=8,n_text_layer=6,n_text_state=512,n_vocab=51865)

class WhisperModelModule(LightningModule):
    def __init__(self,setname,d_model,patch_dim,img_type,audio_features_dir,config, options,model_name="base", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = options
        self.model = multimodal_whisper.load_pretrained_whisper(patch_dim,model_name)
        # self.model = multimodal_whisper.Multimodal_Whisper(testdim,256)#N_MELS, MAX_MIDI - MIN_MIDI + 3, MODEL_COMPLEXITY).to(DEFAULT_DEVICE)
        self.tokenizer = multimodal_whisper.tokenizer.get_tokenizer(True, language=self.options.language, task=self.options.task)
        self.audio_features_dir=audio_features_dir
        self.img_type=img_type
        # both encode decode train
        for p in self.model.decoder.parameters():
            p.requires_grad = False
        summary(self.model)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.__train_dataset,self.__eval_dataset=split_dataset(setname)
        self.setname=setname
        


    # def forward(self, x):
    #     return self.model(x)

    def training_step(self, batch, batch_id):#这里的batch来自Collator
        input_ids = batch["input_ids"]#[16, 80, 3000];mel 80通道 3000个窗口
        labels = batch["labels"].long()#[16, 125];[batch_size,seq_len]，eot，-100。。
        dec_input_ids = batch["dec_input_ids"].long()#eot,eot
        image_features=batch["image_features"]#100,256
        # out = self.model(input_ids, dec_input_ids)#dec_input_ids的预测右移的logits[16, 125, 51865]输入加上sot且扩展eot的token_id;和encoder输出
        out = self.model(image_features,input_ids, dec_input_ids)#dec_input_ids的预测右移的logits[16, 125, 51865]输入加上sot且扩展eot的token_id;和encoder输出
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#交叉熵

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        f=open("checkpoint/checkpoint/log.txt","a")
        f.write("train/loss"+str(int(loss))+'\n')
        # out1=out.clone()
        # labels1=labels.clone()
        # out1[out1 == -100] = self.tokenizer.eot#扩展eot_id
        # labels1[labels1 == -100] = self.tokenizer.eot
        # for i in range(len(batch['input_ids'])):
        #     ground_truth=self.tokenizer.decode(labels1[i])
        #     o = torch.argmax(out1[i], dim=1)
        #     text_using_dec_input_id=self.tokenizer.decode(o)
        #     text_decode=self.model.decode(input_ids[i]).text
        #     image_id=batch['image_ids'][i]
        #     # set the file paths
        #     filename = "result_multimodal_whiper_train_dataset.csv"
        #     # image_folder = "image/"
        #     # audio_folder = "audio/"
        #     data_folder = "data_image_audio/"
            
        #     image_path=os.path.join('data_narrative/images',self.setname,batch['image_ids'][i]+'.jpg')
        #     ogg_file_path=os.path.join('data_narrative/ogg',self.setname,batch['image_ids'][i] +'.ogg')
            
        #     # create the CSV file and write the header row
        #     with open(filename, mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(['image_id', 'ground_truth','text_using_dec_input_id', 'text_decode', 'audio_url'])

        #         # write each line to the CSV file and save image and audio
        #         writer.writerow([batch['image_ids'][i] , ground_truth,text_using_dec_input_id, text_decode, batch['url'][i] ])

        #         # save the image
        #         with open(image_path, 'rb') as image_file:
        #             image_bytes = image_file.read()
        #             with open(data_folder + batch['image_ids'][i] + '.jpg', 'wb') as audio_file:
        #                 audio_file.write(image_bytes)
        #         # save the audio
        #         with open(ogg_file_path, 'rb') as ogg_file:
        #             ogg_bytes = ogg_file.read()
        #             with open(data_folder + batch['image_ids'][i] + '.ogg', 'wb') as audio_file:
        #                 audio_file.write(ogg_bytes)
        return loss
    
    def validation_step(self, batch, batch_id):#5.每训练完一批后，使用验证集获得Loss,cer,wer,计入日志
        input_ids = batch["input_ids"]#[16, 80, 3000];mel 80通道 3000个窗口
        labels = batch["labels"].long()#[16, 125];[batch_size,seq_len]，eot，-100。。
        dec_input_ids = batch["dec_input_ids"].long()#eot,eot
        image_features=batch["image_features"]#[batch,100,256]
        out = self.model(image_features,input_ids, dec_input_ids)#dec_input_ids的预测右移的logits[16, 125, 51865]输入加上sot且扩展eot的token_id;和encoder输出
        # out = self.model(input_ids, dec_input_ids)#dec_input_ids的预测右移的logits[16, 125, 51865]输入加上sot且扩展eot的token_id;和encoder输出

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#交叉熵[1440, 51865];[1440]

        out[out == -100] = self.tokenizer.eot#扩展eot_id
        labels[labels == -100] = self.tokenizer.eot
        #o:<lang>,<>,<notimestamps>
        o_list, l_list = [], []
        for o, l in zip(out, labels):#循环16次
            o = torch.argmax(o, dim=1)#贪婪法预测的token_id_seq
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))#生成text
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
            
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)#记录日志
        f=open("checkpoint/checkpoint/log.txt","a")
        f.write("val/loss"+str(int(loss))+'\n'+"val/cer"+str(cer)+'\n'+"val/wer"+str(wer)+'\n')
        # for i in range(len(batch['input_ids'])):
        #     ground_truth=self.tokenizer.decode(labels[i])
        #     o = torch.argmax(out[i], dim=1)
        #     text_using_dec_input_id=self.tokenizer.decode(o)
        #     text_decode=self.model.decode(input_ids[i]).text
        #     image_id=batch['image_ids'][i]
        #     # set the file paths
        #     filename = "result_multimodal_whiper_val_dataset.csv"
        #     # image_folder = "image/"
        #     # audio_folder = "audio/"
        #     data_folder = "data_image_audio/"
            
        #     image_path=os.path.join('data_narrative/images',self.setname,batch['image_ids'][i]+'.jpg')
        #     ogg_file_path=os.path.join('data_narrative/ogg',self.setname,batch['image_ids'][i] +'.ogg')
            
        #     # create the CSV file and write the header row
        #     with open(filename, mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(['image_id', 'ground_truth','text_using_dec_input_id', 'text_decode', 'audio_url'])

        #         # write each line to the CSV file and save image and audio
        #         writer.writerow([batch['image_ids'][i] , ground_truth,text_using_dec_input_id, text_decode, batch['url'][i] ])

        #         # save the image
        #         with open(image_path, 'rb') as image_file:
        #             image_bytes = image_file.read()
        #             with open(data_folder + batch['image_ids'][i] + '.jpg', 'wb') as audio_file:
        #                 audio_file.write(image_bytes)
        #         # save the audio
        #         with open(ogg_file_path, 'rb') as ogg_file:
        #             ogg_bytes = ogg_file.read()
        #             with open(data_folder + batch['image_ids'][i] + '.ogg', 'wb') as audio_file:
        #                 audio_file.write(ogg_bytes)
        f=open("checkpoint/checkpoint/log.txt","a")
        f.write("val/loss"+str(loss.item())+'\n'+"val/cer"+str(cer)+'\n'+"val/cer"+str(wer)+'\n')
        if batch_id==0:
            self.val_loss=loss
            self.val_cer=cer
            self.val_wer=wer
        elif batch_id==4187-1:
            self.log("val/loss_ep", self.val_loss, on_step=True, prog_bar=True, logger=True)#记录日志
            self.log("val/cer_ep", self.val_cer, on_step=True, prog_bar=True, logger=True)#记录日志
            self.log("val/wer_ep", self.val_wer, on_step=True, prog_bar=True, logger=True)#记录日志
            # with open('checkpoint/checkpoint/eval_result.txt','a') as f:
            #     f.write('------------------------------------------------------ \n')
            #     f.write(str(((self.val_loss+loss)/4187).item())+'\n')
            #     f.write(str((self.val_cer+cer)/4187)+'\n')
            #     f.write(str((self.val_wer+wer)/4187)+'\n')
        else:
            self.val_loss+=loss
            self.val_cer+=cer
            self.val_wer+=wer            

        return {
            "cer": cer,#0.058-0.19
            "wer": wer,#0.02-0.27
            "loss": loss#0.265-0.7
        }

    def configure_optimizers(self):#2.设置优化
        model = self.model
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]#优化的参数
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.config["learning_rate"], 
                        eps=self.config["adam_epsilon"]
                    )#学习率，优化参数，种类
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config["warmup_steps"], 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):#1

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )#2000/16/epoch_num
    
    def train_dataloader(self):
        dataset = WhisperASRDataset(self.__train_dataset, self.tokenizer,self.audio_features_dir,self.img_type,self.setname)
        return torch.utils.data.DataLoader(dataset, 
                    batch_size=self.config["batch_size"], 
                    drop_last=True, shuffle=True, num_workers=self.config["num_worker"],
                    collate_fn=WhisperASRDataCollator()
                )

    def val_dataloader(self):#3.val
        dataset = WhisperASRDataset(self.__eval_dataset, self.tokenizer,self.audio_features_dir,self.img_type,self.setname)#g
        return torch.utils.data.DataLoader(dataset, 
                    batch_size=self.config["batch_size"], 
                    num_workers=self.config["num_worker"],
                    collate_fn=WhisperASRDataCollator()
                )
    
