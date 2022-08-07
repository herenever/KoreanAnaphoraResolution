import json
import csv
import tokenizers
from transformers import ElectraTokenizer
import numpy as np
import pandas as pd
PATH = "/home/jinwon/lab/anaphora/anaphora_dataset/SXZA2002003180.json"

def readJson(path = PATH):
    a_cnt = 0
    n_cnt = 0
    all_cnt = 0
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    with open(path,'r',encoding='UTF8') as f:
        json_data = json.load(f)
        documents = json_data['document']
        for document in documents:
            sentences = document['sentence']
            ZAs = document['ZA']
            for ZA in ZAs:
                predicate = ZA['predicate']
                antecedents = ZA['antecedent']
                antecedent = None
                print(f"predicate : {predicate['form']}")
                for elem in antecedents:
                    if elem['type'] == 'subject':
                        antecedent = elem
                        break
                
                if antecedent == None:
                    continue
            
                predicate_sentence_id = predicate['sentence_id']
                antecedent_sentence_id = antecedent['sentence_id']
                begin = 0
                end = 0
                predicate_sentence = ""
                window_size_sentence = ""
                length = 0
                antecedent_word = ""
                if antecedent_sentence_id == '-1':
                    if n_cnt <=3000:
                        cnt = 0
                        while(sentences[cnt]['id'] != predicate_sentence_id):
                            cnt+=1
                        predicate_sentence = sentences[cnt]['form']
                        window_size_sentence = sentences[cnt-2]['form'] + " " + sentences[cnt-1]['form']
                        length = len(predicate_sentence+window_size_sentence) + 2
                        begin = 0
                        end = 0
                        antecedent_word ="불필요"
                        n_cnt+=1
                else:
                    begin = antecedent['begin']
                    end = antecedent['end']
                    antecedent_word = antecedent['form']
                    cnt = 0
                    while(sentences[cnt]['id'] != predicate_sentence_id):
                        cnt += 1
                    predicate_sentence = sentences[cnt]['form']
                    window_size_sentence = ""
                    
                    if sentences[cnt-2]['id'] != antecedent_sentence_id and sentences[cnt-1]['id'] != antecedent_sentence_id:
                        # re_cnt = 0
                        # while(sentences[re_cnt]['id'] != antecedent_sentence_id):
                        #     re_cnt += 1
                        # if re_cnt >= cnt:
                        #     continue
                        # window_size_sentence = sentences[re_cnt]['form'] + " " + sentences[re_cnt+1]['form']
                        # begin = int(begin) + len(predicate_sentence) + 1
                        # end = int(end) + len(predicate_sentence) + 1
                        # a_cnt +=1
                        continue
                    elif sentences[cnt-2]['id'] == antecedent_sentence_id:
                        begin = int(begin) + len(predicate_sentence) + 1
                        end = int(end) + len(predicate_sentence) + 1
                        window_size_sentence = sentences[cnt-2]['form'] + " "  + sentences[cnt-1]['form']
                        a_cnt +=1
                    elif sentences[cnt-1]['id'] == antecedent_sentence_id:
                        begin = int(begin) + len(predicate_sentence) + len(sentences[cnt-2]['form']) + 2
                        end = int(end) + len(predicate_sentence) + len(sentences[cnt-2]['form']) + 2
                        window_size_sentence = sentences[cnt-2]['form'] + " "  + sentences[cnt-1]['form']
                        a_cnt +=1
                    else:
                        continue
                    length = len(predicate_sentence + window_size_sentence ) + 2
                    
                answer = (begin,end)
            
                if predicate_sentence !="" and window_size_sentence != "" and length <= 500:
                    all_cnt +=1
                    with open('./anaphora_dataset/new_dataset2.csv','a') as cf:
                        wr = csv.writer(cf)
                        wr.writerow([predicate_sentence,window_size_sentence,answer,antecedent_word,length])
                    cf.close()
    f.close()
    print(f"복원 가능 : {a_cnt}")
    print(f"복원 불가능 : {n_cnt}")
    print(f"{all_cnt}")
                
                
f = open('./anaphora_dataset/new_dataset2.csv','w',newline='')
wr = csv.writer(f)
wr.writerow(["document1","document2","label","antecedent","length"])
f.close()        
readJson()
data = pd.read_csv('./anaphora_dataset/new_dataset2.csv',encoding='UTF8')
data['split'] = np.random.randn(data.shape[0],1)

msk = np.random.rand(len(data)) <= 0.8

train = data[msk]
validation = data[~msk]

train.to_csv('./anaphora_dataset/w2_train2.csv',index=False)
validation.to_csv('./anaphora_dataset/w2_validation2.csv',index=False)
                
                
                