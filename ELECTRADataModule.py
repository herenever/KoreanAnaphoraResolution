import os
import re

import numpy
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import transformers
from transformers import ElectraModel,ElectraConfig

from KoCharELECTRA.tokenization_kocharelectra import KoCharElectraTokenizer
from ElectraForFinetuning import ElectraforFinetune
from ElectraAnaphoraResolution import ElectraForResolution

def processAnswer(answer):
  answer = answer.replace("(","")
  answer = answer.replace(")","")
  answer = answer.split(", ")
  begin = int(answer[0])
  end = int(answer[1])
  if begin != 0 and end != 0:
    begin += 1
    end += 1
  return {
      "begin" : [begin],
      "end" : [end]
  }
  
class ResolutionDataset(Dataset):
    def __init__(self,path,doc1_col,doc2_col,label_col,ante_col,max_length,num_wokers=1):
        self.tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
        self.max_length = max_length
        self.doc1_col = doc1_col
        self.doc2_col = doc2_col
        self.label_col = label_col
        self.ante_col = ante_col
        df = pd.read_csv(path,index_col=False)
        df = df.dropna(axis=0)
        df.drop_duplicates(subset=[self.doc1_col],inplace=True)
        self.dataset = df
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        document1 = self.dataset[self.doc1_col].iloc[idx]
        document2 = self.dataset[self.doc2_col].iloc[idx]
        label = self.dataset[self.label_col].iloc[idx]
        ante = self.dataset[self.ante_col].iloc[idx]
        
        answer = processAnswer(label)
        begin = answer['begin']
        end = answer['end']
        
        inputs = self.tokenizer.encode_plus(
            text = document1,
            text_pair = document2,
            add_special_tokens = True,
            pad_to_max_length = True,
            max_length = self.max_length,
            truncation_strategy = "longest_first"
        )
        
        input_ids = torch.LongTensor(inputs['input_ids'])
        token_type_ids = torch.LongTensor(inputs['token_type_ids'])
        attention_mask = torch.LongTensor(inputs['attention_mask'])
        begin = torch.tensor(begin)
        end = torch.tensor(end)
        
        return {
            'input_ids' : input_ids,
            'token_type_ids' : token_type_ids,
            'attention_mask' : attention_mask,
            'begin' : begin,
            'end' : end,
            'ante' : ante
        }
        
class ResolutionDataModule(pl.LightningDataModule):
    def __init__(self,train_path,valid_path,max_length,batch_size,doc1_col,doc2_col,label_col,ante_col,num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.max_length = max_length
        self.doc1_col = doc1_col
        self.doc2_col = doc2_col
        self.label_col = label_col
        self.ante_col = ante_col
        self.num_workers = num_workers
        
    def setup(self,stage=None):
        self.set_train = ResolutionDataset(self.train_path,doc1_col=self.doc1_col,doc2_col=self.doc2_col,label_col=self.label_col,ante_col=self.ante_col,max_length=self.max_length)
        self.set_valid = ResolutionDataset(self.valid_path,doc1_col=self.doc1_col,doc2_col=self.doc2_col,label_col=self.label_col,ante_col=self.ante_col,max_length=self.max_length)
        
    def train_dataloader(self):
        train = DataLoader(self.set_train,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True)
        return train
    
    def val_dataloader(self):
        val = DataLoader(self.set_valid,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False)
        return val
    
    def test_dataloader(self):
        test = DataLoader(self.set_valid,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False)
        return test
    

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    dm = ResolutionDataModule(batch_size=8,train_path="./anaphora_dataset/train_V1.csv",valid_path="./anaphora_dataset/validation_V1.csv",max_length=258,doc1_col='document1',doc2_col='document2',label_col='label',ante_col='antecedent')
    dm.setup()
    tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
    t = dm.train_dataloader()
    v = dm.val_dataloader()
    for idx,data in enumerate(t):
        if idx >50:
            break
        # begin = data['begin'].squeeze(-1).tolist()
        # end = data['end'].squeeze(-1).tolist()
        # pair = list(zip(begin,end))
        # for truep,inp in zip(pair,data['input_ids']):
        #     b,e = truep
        #     result = tokenizer.decode(inp[b:e])
        #     print(result)
