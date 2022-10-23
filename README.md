# Korean Anaphora Resolution
한국어 생략어 복원 모델
***
## Base

<img width="446" alt="diagram" src="https://user-images.githubusercontent.com/74087118/197397920-7c71f4e4-0493-4980-b801-9d151d3c1f6c.PNG">

+ 한국어 생략어(주어) 복원 모델입니다.
+ 사전 학습된 KoCharElectra LM에 BI-lstm layer를 쌓아 Finetuning 시킨 모델입니다.
+ 학습에 사용된 데이터
  + 국립국어원 모두의 말뭉치의 **무형 대용어 복원 말뭉치 2020**
  + 진술서 내용 중 일부를 파싱하여 제작된 데이터
+ WS를 2 문장 이내로 학습시켰습니다.
+ validation Acc 는 80.58% 입니다.

***
## Usage
+ Resolution.py의 AnaphoraResolution 객체를 생성하고 resolution method를 호출하여 사용합니다.

```python
# example
from Resolution import AnaphoraResolution

ar = AnaphoraResolution()
pre = "그래서 학교에서 밤을 샜다."
suf = "진원이는 해야 할 일들이 많이 있었다."
result = ar.resolution(pre,suf)
print(f"복원 전: {suf} {pre}")
print(f"복원 후: {suf} {result}(이/가) {suf}")

# output
# 복원 전: 진원이는 해야 할 일들이 많이 있었다. 그래서 학교에서 밤을 샜다.
# 복원 후: 진원이는 해야 할 일들이 많이 있었다. 진원(이/가) 그래서 학교에서 밤을 샜다.
```

***
## References
<https://github.com/monologg/KoCharELECTRA>
