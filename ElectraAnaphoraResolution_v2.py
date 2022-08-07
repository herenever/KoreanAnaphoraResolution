from transformers import ElectraConfig, ElectraPreTrainedModel, ElectraModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import loggers as pl_logger
from ElectraAnaphoraResolution import ElectraForResolution

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
    
class ElectraForResolution_v2(pl.LightningModule):
    def __init__(self,learning_rate):
        super().__init__()
        self.electra = ElectraForResolution.load_from_checkpoint("/home/jinwon/lab/anaphora/model_checkpoint/version_2/epoch=46--total_Accuracy_Val=0.7710.ckpt")
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(
            self,
            input_ids =None,
            attention_mask = None,
            token_type_ids = None,
            start_positions = None,
            end_positions = None
        ):
        outputs = self.electra(input_ids=input_ids,attention_mask = attention_mask,token_type_ids=token_type_ids,start_positions=start_positions,end_positions =end_positions)
        loss_score = None
        try:
            loss_score,start_logits,end_logits = outputs[:3]
        except:
            start_logits,end_logits = outputs[:2]
            
        weight = [[[1.6] + [0.0] * (len(token_type_ids[0])-1)] for _ in range(len(token_type_ids))]
        weight = torch.Tensor(weight).view(len(token_type_ids),len(token_type_ids[0])).to(device)
        
        weight = weight + token_type_ids
        start_logits = start_logits * weight
        end_logits = end_logits * weight
        
        outputs = (start_logits,end_logits,)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignore_index = start_logits.size(1)
            start_positions.clamp_(0,ignore_index)
            end_positions.clamp_(0,ignore_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index = ignore_index)
            start_loss = loss_fct(start_logits,start_positions)
            end_loss = loss_fct(end_logits,end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
    
    def training_step(self,batch,batch_idx):
        begin = batch['begin'].squeeze(1)
        end = batch['end'].squeeze(1)
        output =self(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device),
            token_type_ids = batch['token_type_ids'].to(device),
            start_positions = begin.to(device),
            end_positions = end.to(device)
        )
        loss,begin_score,end_score = output[:3]
        pred_begin = nn.functional.softmax(begin_score,dim=-1).argmax(dim=-1)
        pred_end = nn.functional.softmax(end_score,dim=-1).argmax(dim=-1)
        true_antecedent = batch['ante'] # list[8]
        input_ids = batch['input_ids']
        
        self.log("Loss/train_step",loss,on_step=True)
        
    
        return {
            'loss' : loss,
            'pred_begin' : pred_begin,
            'pred_end' : pred_end,
            'true_antecedent' : true_antecedent,
            'input_ids' : input_ids,
            'begin' : begin,
            'end' : end
        }
        
    def training_epoch_end(self, outputs,state="train"):
        y_pred = []
        y_true = []
        loss = 0
        cnt = 0
        
        b_match = 0
        e_match = 0
        total = 0
        
        for batch in outputs:
            cnt+=1
            loss+=batch['loss']
            pred_begin = batch['pred_begin'].tolist()
            pred_end = batch['pred_end'].tolist()
            pred_pair = list(zip(pred_begin,pred_end))
            true_begin = batch['begin'].tolist()
            true_end = batch['end'].tolist()
            true_pair = list(zip(true_begin,true_end))
            total += len(pred_pair)
            for true,pred in zip(true_pair,pred_pair):
                tb,te = true
                pb,pe = pred
                if tb == pb :
                    b_match += 1
                if te == pe:
                    e_match += 1
        
        begin_acc = b_match / total
        end_acc = e_match /total
        total_acc = (begin_acc+end_acc)/2
                    
                
        self.log("Loss/Train",loss,on_epoch=True)
        self.log("total_Accuracy/Train",total_acc,on_epoch=True)
        self.log("begin_Accuracy/Train",begin_acc,on_epoch=True)
        self.log("end_Accuracy/Train",end_acc,on_epoch=True)
        
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] total_Accuracy : {total_acc:.4f} begin_Accuracy : {begin_acc:.4f} end_Accuracy : {end_acc:.4f}')
    
    def validation_step(self,batch,batch_idx):
        y_true = []
        y_pred = []
        total = 0
        b_match = 0
        e_match = 0
        begin = batch['begin'].squeeze(1)
        end = batch['end'].squeeze(1)
        output = self(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device),
            token_type_ids = batch['token_type_ids'].to(device)
        )
        begin_score,end_score = output[:2]
        pred_begin = nn.functional.softmax(begin_score,dim=-1).argmax(dim=-1)
        pred_end = nn.functional.softmax(end_score,dim=-1).argmax(dim=-1)
        pred_pair = list(zip(pred_begin,pred_end))
        true_begin = begin.tolist()
        true_end = end.tolist()
        true_pair = list(zip(true_begin,true_end))
        total += len(pred_pair)
        for true,pred in zip(true_pair,pred_pair):
            tb,te = true
            pb,pe = pred
            if tb == pb :
                    b_match += 1
            if te == pe:
                e_match += 1
        
        begin_acc = b_match / total
        end_acc = e_match /total
        total_acc = (begin_acc+end_acc)/2
            
        
        # input_ids = batch['input_ids']
        # true_antecedent = batch['ante']
        # for pair,input_id,ante in zip(pred_pair,input_ids,true_antecedent):
        #     begin,end = pair
        #     pred =''
        #     if begin == 0 and end == 0:
        #         pred = '불필요'
        #     else:
        #         pred = self.tokenizer.decode(input_id[begin:end+1])
        #     ante = normalize(ante)
        #     pred = normalize(pred)
        #     y_true.append(ante)
        #     y_pred.append(pred)
            
        # result = self.categorization(y_true,y_pred)
        # true_positive, false_positive, true_negative, false_negative = result
        # acc = self.accuracy(true_positive, false_positive, true_negative, false_negative)
        # pre = self.pre(true_positive, false_positive)
        # rec = self.recall(true_positive, false_negative)
        # f1 = self.f1(true_positive, false_positive, false_negative)
        
        self.log("total_Accuracy_Val",total_acc,on_epoch=True)
        self.log("begin_Accuracy_Val",begin_acc,on_epoch=True)
        self.log("end_Accuracy_Val",end_acc,on_epoch=True)
        # self.log('val_accuracy',acc,on_epoch=True,prog_bar=True)
        # self.log('val_precision',pre,on_epoch=True,prog_bar=True)
        # self.log('val_recall',rec,on_epoch=True,prog_bar=True)
        # self.log('val_f1',f1,on_epoch=True,prog_bar=True)
        
        return {
            'total_acc' : total_acc,
            'begin_acc' : begin_acc,
            'end_acc' : end_acc
        } 
        
    def validation_epoch_end(self, outputs):
        total_acc = [i['total_acc'] for i in outputs]
        begin_acc = [i['begin_acc'] for i in outputs]
        end_acc = [i['end_acc'] for i in outputs]
        # f1 = [i['val_f1'] for i in outputs]
        
        total_acc = np.mean(total_acc)
        begin_acc = np.mean(begin_acc)
        end_acc = np.mean(end_acc)
        # f1 = np.mean(f1)
        
        # print(f'[VALIDATION] val_accuracy : {acc:.4f} val_precision : {pre:.4f} val_recall : {rec:.4f} val_f1_score : {f1:.4f}')
        print(f'[VALIDATION] total_Accuracy : {total_acc:.4f} begin_Accuracy : {begin_acc:.4f} end_Accuracy : {end_acc:.4f}')
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.electra.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler
        }