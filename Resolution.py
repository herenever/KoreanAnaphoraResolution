from ELECTRADataModule import *
from ElectraAnaphoraResolution import *


class AnaphoraResolution:
    def __init__(self):
        self.tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
        self.model = ElectraForResolution.load_from_checkpoint("./model_checkpoint/epoch=99--total_Accuracy_Val=0.8058.ckpt")
        self.max_length = 256
        
    def resolution(self,pre:str,suf:str):
        r"""
            pre : 생략어 발생 문장,
            suf : 선행 문장들
        """
        prefix = pre
        suffix = suf
        inputs = self.tokenizer.encode_plus(
            text = prefix,
            text_pair = suffix,
            add_special_tokens = True,
            pad_to_max_length=True,
            max_length= self.max_length
            
        )
        input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0)
        token_type_ids = torch.LongTensor(inputs['token_type_ids']).unsqueeze(0)
        attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0)
        
        output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        start,end = output
        start  = nn.functional.softmax(start,dim=-1).argmax()
        end = nn.functional.softmax(end,dim=-1).argmax()
        result = self.tokenizer.decode(input_ids[0][start:end])
        result = self.normalize(str(result))
        return result
        
    def normalize(self,word:str):
        word = word
        special_token = ['[CLS]','[UNK]','[PAD]','[SEP]','[MASK]']
        for token in special_token:
            word.replace(token,"")
        # TODO: pos tag 사용해서 NP(순수 단어만 추출해야하는데 issue가 있어서 일단은 제거)
        word = word.strip()
        return word
        
if __name__ == "__main__":
    ar = AnaphoraResolution()
    resolution = ar.resolution("생략어 발생 문장!","선행 문장들!")
    test = ar.resolution("그상태에서 민정이를 부른거예요 이리오라고","그러니까 김진수 오빠가 이우진 오빠 집에 있었는데")
    print(test)
    # return 값은 생략된 선행어 inference 값
    # input으로 특정한 특수문자 들어가면 UNK 토큰으로 복원될 가능성 있음 => vocab에 특수문자 추가하면 될 듯?
    