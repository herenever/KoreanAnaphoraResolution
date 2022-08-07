from base64 import encode
import csv
import pandas as pd

def convertDataframe(path,name):
    df = pd.read_csv(path,index_col=False,encoding='CP949')
    doc1 = df['document1']
    doc2 = df['document2']
    ante = df['antecedent']
    length = len(doc1)
    with open("./anaphora_dataset/"+name+'.csv','a') as f:
        wr = csv.writer(f)
        wr.writerow(["document1","document2","label","antecedent","length"])
        for i in range(length):
            query_len = len(doc1.iloc[i].strip())
            doc2sen = doc2.iloc[i].strip()
            word = str(ante.iloc[i]).strip()
            begin,end = searchInSentence(doc2sen,word)
            if begin == -1 and end == -1:
                begin,end = 0,0
            else:
                begin += (query_len + 1)
                end += (query_len + 1)
            answer = (begin,end)
            wr.writerow([doc1.iloc[i].strip(),doc2.iloc[i].strip(),answer,ante.iloc[i],len(doc1.iloc[i].strip()+doc2.iloc[i].strip())])
    f.close()
            

        
def searchInSentence(sentence,word):
    idx = 0
    word_len = len(word)
    flag = False
    for i in range(len(sentence)):
        if sentence[i:i+word_len] == word:
            idx = i
            flag = True
            break
    if flag:return idx,idx+word_len
    else: return -1,-1


convertDataframe("/home/jinwon/lab/anaphora/anaphora_dataset/final_data.csv","statement")
# print(searchInSentence("안녕하세요 저는 홍진원입니다.","홍진원"))
        
        
        

