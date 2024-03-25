import openai
import pandas as pd
from tqdm import tqdm
import argparse
from konlpy.tag import Mecab
import time
from compute_f1score import process_data_and_evaluate

def open_text(file_path):
    with open(file_path, 'r') as file:
        defn = file.read()
    return defn

def open_csv(file_path):
    df = pd.read_csv(file_path,encoding='cp949',index_col=0)
    return df

def word_pos_match(sentence):
    mecab = Mecab()
    t = sentence.split(" ")
    t_s = mecab.pos(sentence)
    t_dict = {i:[] for i in range(len(t))}

    cnt = 0
    for i in range(len(t)):
        tmp_text = ""
        while cnt < len(t_s):
            if tmp_text != t[i]:
                if t_s[cnt][0] in t[i]:
                    tmp_text += t_s[cnt][0]
                    t_dict[i].append(t_s[cnt])
                    cnt += 1
                else:
                
                    break
            else:
                break
    return t_dict

# 질의 폼 형성
def query_format(file_path):
    
    input_data = open_csv(file_path)
    
    def format_structure(sentence, pos):
        text = []
        s = sentence.split(" ")
        match_dict = word_pos_match(sentence) # 공백단위 단어 분절과 포스태깅 결과 매칭 데이터

            
        text.append('Word | POS | Entity | BIO | Thought ')

        for i in range(len(s)):
            match_pos = "+".join([f"{token[0]}({token[1]})" for token in match_dict[i]])
            check_pos = [token[1] for token in match_dict[i]]

            text.append(f"{s[i]} |{match_pos}|  |  |  ")
            
        return "\n".join(text)

    input_data['query_format'] = input_data.apply(lambda row: format_structure(row['sentences'], row['POS']), axis=1)

    return input_data

# ChatGPT 질의
def gpt_qa(args, defn, input_data):
    model = args.gpt_model
    k = args.k_sample
    start_num = args.start_num
    end_num = args.end_num
    answers = []
    if start_num == 0:
        input_data[f'{k}-Shot result'] = ""
    print("start_num: ", start_num)
    print("end_num: ", end_num)

    for cnt in tqdm(range(start_num,end_num)):
        query = defn
        user_messages = []

        # System message
        system_message = {"role": "system", "content": "You are the most helpful assistant in the field of Named Entity Recognition in the Korean healthcare domain."
                          + "Write the Thought and BIO tags for the questions by referring to the following examples."}
        user_messages.append(system_message)

        # k-Shot Samples
    
        for i in range(k):
            user_message = {"role": "user", "content": input_data.iloc[cnt][f'{i+1}-Shot']}
            user_messages.append(user_message)

        # Input sentence
        input_sentence = f"Input:\n- input sentence: {input_data.iloc[cnt]['sentences']}\n\nOutput:\n{input_data.iloc[cnt]['query_format']}"
        user_message = {"role": "user", "content": input_sentence}
        user_messages.append(user_message)

        # Concatenate user messages
        for message in user_messages:
            query += f"\n{message['content']}"

        # for i in query.split('\n'):
        #     print(i)

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(
            model=model,
            messages=user_messages,
            temperature = 0
        )

        answer = response['choices'][0]['message']['content']
        input_data[f'{k}-Shot result'][cnt] = answer

        # saveing result #
        input_data.to_csv(f'./result/{k}-shot/{args.gpt_model}/{k}-shot_current_result.csv', encoding='cp949')

        input_csv = f'./result/{k}-shot/{args.gpt_model}/{k}-shot_current_result.csv'
        output_csv = f"./result/{k}-shot/{args.gpt_model}/{k}-shot_tag_info.csv"
        output_json = f'./result/{k}-shot/{args.gpt_model}/{k}-shot_evaluation_result.json'
        answers.append(answer)
        try:
            process_data_and_evaluate(k, input_csv, output_csv, output_json)
        except:
            continue

        

    input_data.to_csv(f'./result/{k}-shot/{args.gpt_model}/{k}-shot_result.csv', encoding='cp949')
    return input_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPENAI_API_KEY", type=str, default= "")
    # parser.add_argument("--train_file", type=str, default= "./data/train_data.csv")
    # parser.add_argument("--eval_file", type=str, default= "./data/test_data.csv")
    parser.add_argument("--test_file", type=str, default= "../9-Shot_result.csv")
    parser.add_argument("--defn_file", type=str, default= "../data/Define.txt")
    parser.add_argument("--gpt_model", type=str, default= "gpt-4")
    parser.add_argument("--start_num", type=int, default= 0)
    parser.add_argument("--end_num", type=int, default= 450)
    parser.add_argument("--k_sample", type=int, default= 5)
    args = parser.parse_args()

    # openai API 키 인증
    openai.api_key = args.OPENAI_API_KEY

    defn = open_text(args.defn_file)
    input_data = query_format(args.test_file)

    result = gpt_qa(args,defn,input_data)
    print(f"---{args.k_sample}-Shot NER DONE ---")
    print(len(result.loc[result[f'{args.k_sample}-Shot result'] != ""]))

    k = args.k_sample

    input_csv = f'./result/{k}-shot/{args.gpt_model}/{k}-shot_current_result.csv'
    output_csv = f"./result/{k}-shot/{args.gpt_model}/{k}-shot_tag_info.csv"
    output_json = f'./result/{k}-shot/{args.gpt_model}/{k}-shot_evaluation_result.json'
    
    process_data_and_evaluate(k, input_csv, output_csv, output_json)