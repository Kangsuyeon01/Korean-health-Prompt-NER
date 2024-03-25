import pandas as pd
import re
from seqeval.metrics import f1_score, classification_report
import json


def str2list(row):
    return row.split(" ")

if __name__ == "__main__":
    df = pd.read_csv('./train_result/all_train/bert-base-uncased/result_1.csv',encoding='cp949',index_col=0)

    # df.dropna(subset=['5-Shot result'], inplace=True)

    # df['predict'] = df['5-Shot result'].apply(split_tag)
    # 예측과 정답 열 데이터
    df['bio_tag'] = df['bio_tag'].apply(str2list)
    df['predict'] = df['predict'].apply(str2list)

    true_tags = df['bio_tag']
    predicted_tags = df['predict']

            # Replace BIO tags in bio_tag and predict columns
    tag_mapping = {
        'B-DS': 'DS_B',
        'I-DS': 'DS_I',
        'B-ST': 'ST_B',
        'I-ST': 'ST_I',
        'B-BD': 'BD_B',
        'I-BD': 'BD_I',
        'O': 'O'
    }

    df['bio_tag'] = df['bio_tag'].apply(lambda tags: [tag_mapping[tag] for tag in tags])
    df['predict'] = df['predict'].apply(lambda tags: [tag_mapping[tag] for tag in tags])

    test_tags = df['bio_tag']
    pred_tags = df['predict']

    # F1 평가 결과
    eval_report = classification_report(test_tags, pred_tags)
    eval_f1_score = "{:.1%}".format(f1_score(test_tags, pred_tags))
    print(eval_report)

    # 결과를 JSON 형식으로 저장
    result = {
        "remain index len": len(df),
        "classification_report": eval_report.split("\n"),
        "f1_score": eval_f1_score
    }
    with open("./train_result/all_train/bert-base-uncased/bert_base_result1.json", 'w') as json_file:
        json.dump(result, json_file, indent=4)

    