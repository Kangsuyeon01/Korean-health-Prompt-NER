# Korean-health-Prompt-NER

- `개발 기간`: 2023.06 ~ 2023.11
- `참여 인원` : 1
- `사용 언어` : Python


한국어 헬스케어 개체명 인식을 위한 거대 언어 모델에서의 형태소 기반 Few-Shot 학습 기법 (2023.11.8) 한국정보처리학회 학술대회논문집, 30(2), 428-429. 


- 본 연구는 한국어 헬스케어 분야의 개체명 인식 정확도를 높이기 위해, 형태소 정보를 활용한 Few-Shot 학습 기법과 거대 언어 모델(LLM)의 프롬프트 엔지니어링을 접목한 방법 제안함
- 한국어의 교착어 특성을 고려하여, BERT와 같은 기존 모델들이 대규모 데이터에 의존하는 한계를 넘어고자 거대언어모델을 통한 개체명인식 방법 제안함
![image](https://github.com/Kangsuyeon01/Korean-health-Prompt-NER/assets/94098065/252fa130-6e79-4c96-81dd-42f1c1a93d26)

## Dataset
- 네이버 지식인 정보를 크롤링하여 사용
- 총 1830개의 문장을 추출하였으며, 개체명의 분류는 질병(DS), 증상(ST), 신체(BD)로 나누어 BIO 태깅 방식 적용

|      | B-DS | I-DS | B-ST | I-ST | B-BD | I-BD |
|------|------|------|------|------|------|------|
| Entity| 818  | 333  | 1076 | 114  | 884  | 37   |

## Method
- Prompt 질의를 위해 기존의 개체명 인식 데이터를 새롭게 구축
![image](https://github.com/Kangsuyeon01/Korean-health-Prompt-NER/assets/94098065/3a6742f4-f72a-43ba-836a-3a6fefbc350f)

- 정의(Definition), 질의(Q),Few-Shot 예시(Sample),입력(Input) 구조
![image](https://github.com/Kangsuyeon01/Korean-health-Prompt-NER/assets/94098065/b8555668-c944-4ddf-ac6e-8b3850630d70)

- 제안하는 Few-Shot 프롬프트 질의 과정
  
![image](https://github.com/Kangsuyeon01/Korean-health-Prompt-NER/assets/94098065/1df6d869-60e7-4949-8113-7a0b7ea9bc6c)


1. __형태소 기반 Few-Shot 학습 및 데이터 구축__
  - 헬스케어 관련 데이터를 수집하기 위해 웹 크롤링을 통해 `질병`, `증상`, `신체 부위` 
    등에 대한 문장들을 추출하여 `BIO 태깅` 진행
  - 형태소 정보를 추가하여, 입력 문장에 대한 Few-Shot 학습 데이터 구축
2. __프롬프트 엔지니어링__
  - __유사도 기반 문서 검색__: `Sentence BERT`를 활용하여 주어진 문장과 유사한 문
    서를 검색하고, 이를 바탕으로 `Few-Shot 프롬프트` 구성
  - __프롬프트 구성__: 정의, 질의, Few-Shot 예시, 입력을  포함하는 구조를 통해, 
    형태소 정보를 포함한 입력 문장을 LLM에 제공하기 위한 프롬프트를 설계
  - __LLM 질의 및 성능 평가__: 구성된 프롬프트를 거대 언어 모델에 질의하여 개체명 
    인식 결과를 얻고, 각 모델 별 및 Shot의 개수 별로 성능 평가를 진행


---
## Result
- 제안하는 프롬프트기반 개체명 인식 기법의 검증을 위해 Open AI에서 발표한 거대 언어 모델인 `gpt-3.5-turbo-16k` 모델과 `gpt-4` 모델을 API를 통해 호출해 사용
- 비교 실험을 위해 사전 학습된 언어 모델인 'bert-base-multilingual-cased'를 활용하여 실험 진행
- 평가에는 각 개체명 태그에 대한 `F1-Score`를 성능에 대한 평가 지표로 사용

|      | bert-base-multilingual-cased | gpt-3.5-turbo-16k (1-shot)  | gpt-3.5-turbo-16k (2-shot) | gpt-3.5-turbo-16k (5-shot) | __gpt-4 (5shot)__|
|------|-------------------------------|-------------------|--------|--------|--------|
| B-DS | 0.75                          | 0.67              | 0.77   | 0.77   | 0.81   |
| I-DS | 0.2                           | 0.22              | 0.26   | 0.34   | 0.3    |
| B-ST | 0.62                          | 0.58              | 0.66   | 0.68   | 0.73   |
| I-ST | 0.63                          | 0.46              | 0.49   | 0.55   | 0.6    | 
| B-BD | 0.79                          | 0.64              | 0.71   | 0.74   | 0.8    | 
| I-BD | 0                             | 0.42              | 0.43   | 0.46   | 0.53   | 
| TOTAL| 70.20%                        | 60.60%            | 67.90% | 70.10% | 75.30% | 

---
## How to run

* 별도의 Open AI API 발급이 필요합니다.
```
git clone https://github.com/Kangsuyeon01/Korean-health-Prompt-NER.git
cd Korean-health-Prompt-NER/openai-access
```
```
python verify_result.py --k_sample=  --OPENAI_API_KEY=
```

## Contribution
한국어의 언어적 특성을 잘 반영하면서, 대량의 데이터에 의존하는 기존 언어 모델의 한계를 극복하는 개체명 인식 방법을 제안하는 것을 핵심 목표로하였습니다. 



