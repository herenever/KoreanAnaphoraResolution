import csv


PATH = "/home/jinwon/lab/anaphora/anaphora_dataset/"
merge_path_train = "/home/jinwon/lab/anaphora/anaphora_dataset/train_V1.csv"
merge_path_valid = "/home/jinwon/lab/anaphora/anaphora_dataset/validation_V1.csv"
train_list = [PATH+'train.csv',PATH+'state_train.csv']
valid_list = [PATH+'validation.csv',PATH+'state_validation.csv']

with open(merge_path_valid,'w') as f:
    for i,file in enumerate(valid_list):
        if i ==0:
            with open(file,'r') as f2:
                while True:
                    line = f2.readline()
                    
                    if not line:
                        break
                    
                    f.write(line)
            f2.close()
        else:
            with open(file,'r') as f2:
                n=0
                while True:
                    line = f2.readline()
                    if not line:
                        break
                    if n!=0:
                        f.write(line)
                    n+=1
            f2.close()
f.close()
        
                    