from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import TokenError
from transformers import ElectraConfig, ElectraPreTrainedModel, ElectraModel
import torch
import torch.nn as nn


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
class ElectraforFinetune(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"
    
    def __init__(self,config):
        super().__init__(config)
        self.num_labes = config.num_labels
        self.electra = ElectraModel(config)
        self.lstm = nn.LSTM(input_size = config.embedding_size,hidden_size=128,num_layers = 2, batch_first=True,bidirectional = True)
        self.qa_outputs = nn.Linear(256,config.num_labels)
        self.post_init()

        
    def forward(
            self,
            input_ids =None,
            attention_mask = None,
            token_type_ids = None,
            start_positions = None,
            end_positions = None
        ):
        discriminator_hidden_states = self.electra(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        sequence_output = discriminator_hidden_states[0]
        sequence_output,(hidden,cell) = self.lstm(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits,end_logits = logits.split(1,dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # print(f"start before : {start_logits[0].tolist()}")
        # print(f"end before : {end_logits[0].tolist()}")
        
        weight = [[[0.3] + [0.0] * (len(token_type_ids[0])-1)] for _ in range(len(token_type_ids))]
        weight = torch.Tensor(weight).view(len(token_type_ids),len(token_type_ids[0])).to(device)
        
        weight = weight + token_type_ids
        start_logits = start_logits * weight
        end_logits = end_logits * weight
        
        # print(f"start after : {start_logits.tolist()}")
        # print(f"end after : {end_logits.tolist()}")

        outputs = (start_logits,end_logits,) + discriminator_hidden_states[1:]
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
    
