import numpy as np
import os
import pandas as pd
import urllib.request
import faiss
import time
import argparse
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd
from konlpy.tag import Mecab

import torch

# Mecab 객체 생성
mecab = Mecab()

# 저장된 임베딩된 데이터 파일을 로드
def open_pickle_file(embedding_file_path):
    with open(embedding_file_path, 'rb') as f:
        loaded_encoded_data = pickle.load(f)
    return loaded_encoded_data

# 저장된 Defn 파일 로드
def load_define(defn_file):
    with open(defn_file, 'r') as file:
        content = file.read()
        lines = content.split('\n')
    return lines

# csv파일 불러오기
def load_input(file_path):
    df = pd.read_csv(file_path,encoding='cp949')
    return df

# input과 유사 문장 찾기
def s_bert_similarity(args,query):
    global encoded_data, model, train,train_sentence
 
    top_k = int(args.k_sample)
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, encoded_data)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k+20]

    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in corpus:")

    samples = [] # 각 쿼리에 대한 샘플 저장
    check = [] # 중복 문장 체크
    for idx in top_results[0:top_k+20]: # k개의 유사한 샘플 저장
        if len(samples) == top_k:
            break
        # print(train_sentence[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
        if train_sentence[idx] not in check:
            train_loc = train.loc[train['sentences'] == train_sentence[idx]]
            top_sample = train_loc.head(1)
            samples.append(top_sample)
            check.append(train_sentence[idx])
    return samples



# 형태소 태깅 함수
def morpheme_tagging(text):
    node = mecab.pos(text)
    node = ["("+ str(i[0]) +','+ str(i[1])+")" for i in node]
    return " / ".join(node)

""" Defn: 개체명 태그는 각각 BodyPart, Symptom, Disease가 있다. 
각각 BIO 태깅 기법으로 각 개체에 대해 BodyPart는 B-BD 와 I-BD, Symptom는 B-SP와 I-SP, Disease는 B-DS와 I-DS로 정의한다.
문장은 공백 단위로 분리한다. BIO가 O인 경우에는 Entity False로 처리하고, 그 외는 True로 처리한다.

Q: 주어지는 문장에 대해 샘플을 참고하여 Input Sentence에대해 BIO 태그를 이용하여 Sample과 동일하게 output을 구성하여라

Sample:
- sample sentence: 어지럼증이 걱정되어 질문하신 것 같습니다.
- sample POS: 어지럼증(NNG) / 이(JKS) / 걱정(NNG) / 되(XSV) / 어(EC) / 질문(NNG) / 하(XSV) / 신(EP+ETM) / 것(NNB) / 같(VA) / 습니다(EF) / .(SF) / 
- sample BIO: B-ST O O O O O
- sample output:
Word | POS | Entity | BIO 
어지럼증이 |  어지럼증(NNG) + 이(JKS) | True | as it is Symptom (B-ST)
걱정되어 | 걱정(NNG) + 되(XSV) + 어(EC) | False | as it is a verb (O)
질문하신 | 질문(NNG) + 하(XSV) + 신(EP+ETM) | False | as it is a verb (O)
것 | 것(NNB) | False | as it is not a named entity (O)
같습니다 | 같(VA) + 습니다(EF) | False| as it is a verb(O)
. | .(SF) | False | as it is a preiod (O) """

def sample_format(args, query):
    samples = s_bert_similarity(args,query)
    # defn = load_define(args.defn_file) # defn 불러오기
    total_sample = [] # + defn[:]
    for i in range(len(samples)):
        samples_document = [] 
        samples_document.append(f"Sample {i+1} :")
        samples_document.append(f"- sample sentence {i+1} : {' '.join(samples[i]['sentences'].values)}")
        # samples_document.append(f"- sample pos {i+1} : {' '.join(samples[i]['pos'].values)}")
        # samples_document.append(f"- sample BIO {i+1} : {' '.join(samples[i]['bio_tag'].values)}")
        samples_document.append(f"- sample output {i+1}")
        sample_format = str(' '.join(samples[i]['format'].values)).split("\n")
        samples_document += sample_format
        s_format = "\n".join(samples_document)
        total_sample.append(s_format)
    # result =  "\n".join(samples_document)
    return total_sample

# Format 구성
def pipeline(args):
    eval_df = load_input(args.eval_file) # input 불러오기
    eval_df['POS'] = eval_df['sentences'].apply(morpheme_tagging)
    # total_sample = eval_df['sentences'].apply(lambda x: sample_format(args,x))
    total_sample = eval_df['sentences'].apply(lambda x: sample_format(args, x))
    for i in range(args.k_sample):
        tmp = [doc[i] for doc in total_sample]
        col_title = str(i+1) + '-Shot'
        eval_df[col_title] = tmp
    
    return eval_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeded_file", type=str, default= "./data/pickle/encoded_train_data.pkl")
    parser.add_argument("--train_list_file", type=str, default= "./data/pickle/train_list_data.pkl")
    parser.add_argument("--train_file", type=str, default= "./data/train_data.csv")
    parser.add_argument("--eval_file", type=str, default= "./data/test_data.csv")
    parser.add_argument("--defn_file", type=str, default= "./data/Define.txt")
    parser.add_argument("--k_sample", type=int, default= 5)
    
    args = parser.parse_args()


    # 센텐스 버트 임베딩
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train = load_input(args.train_file)
    encoded_data = open_pickle_file(args.embeded_file)
    train_sentence = open_pickle_file(args.train_list_file)

    # NumPy 배열을 PyTorch 텐서로 변환 후 GPU로 이동
    encoded_data = torch.tensor(encoded_data, dtype=torch.float32, device=device)

    model = SentenceTransformer("jhgan/ko-sroberta-multitask").to(device)  # 모델을 GPU로 옮김
    print("임베딩 파일 불러오기 완료")

    # 샘플 매칭
    result = pipeline(args)
    result.to_csv(str(args.k_sample) + "-Shot_result.csv", encoding='cp949')


    
    
