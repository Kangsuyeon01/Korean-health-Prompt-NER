import pandas as pd
import re
from seqeval.metrics import f1_score, classification_report
import json

def split_tag(predict):
    try:
        result = predict.split('\n')
        predict_tags = []
        for i in range(1,len(result)):

            pred = result[i].split('|')[-2]
            pred = pred.replace(" ", "")

            tags = ['B-DS','I-DS',"B-ST","I-ST","B-BD","I-BD","O"]            
            
            if pred not in tags or pred == "":
                pred = "O"
            predict_tags.append(pred)

        return predict_tags # GPT 를 통해 받은 결과 값
    except:
        print(predict)

def str2list(row):
    return row.split(" ")
def process_data_and_evaluate(k, input_csv, output_csv, output_json):
    df = pd.read_csv(input_csv, encoding='cp949', index_col=0)
    df.dropna(subset=[f'{k}-Shot result'], inplace=True)
    df['predict'] = df[f'{k}-Shot result'].apply(split_tag)

    df.dropna(subset=['predict'], inplace=True)

    df['bio_tag'] = df['bio_tag'].apply(str2list)
    true_tags = df['bio_tag']
    predicted_tags = df['predict']

    miss = df[df['bio_tag'].apply(len) != df['predict'].apply(len)]
    df = df[~df.index.isin(miss.index)]

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
        "wrong match sentence": miss.index.tolist(),
        "remove index len": len(miss),
        "remain index len": len(df),
        "classification_report": eval_report.split("\n"),
        "f1_score": eval_f1_score
    }

    # CSV 파일 저장
    # df[['sentences', 'bio_tag', 'predict']].to_csv(output_csv, encoding='cp949')

    # JSON 파일 저장
    with open(output_json, 'w') as json_file:
        json.dump(result, json_file, indent=4)

if __name__ == "__main__":
    k = 5
    input_csv = f'./result/{k}-shot/gpt-4/{k}-shot_current_result.csv'
    output_csv = f"./result/{k}-shot/gpt-4/{k}-shot_tag_info_detail.csv"
    output_json = f'./result/{k}-shot/gpt-4/{k}-shot_evaluation_result_detail.json'
    
    process_data_and_evaluate(k, input_csv, output_csv, output_json)