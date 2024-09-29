import csv
import os
import sys
from openai import OpenAI
from tqdm import tqdm  # tqdmをインポート

# 自作クラスのインポート
from classes.meeting import Meeting
from classes.dialogue_turn import DialogueTurn
from classes.result import OneTurnResult
from classes.result import Result

# ファイルのパス
dir_path = os.path.dirname(os.path.abspath(__file__))
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_path = os.path.join(dir_path, 'result')
result_json_path = os.path.join(dir_path, 'result_json')

#実験設定
MODEL = "gpt-4o-mini"
METHOD = "baseline"

# csv_topics_path内のすべてのCSVファイルのパスを取得
def get_csv_files(csv_topics_path):
    csv_files = [f for f in os.listdir(csv_topics_path) if f.endswith('.csv')]
    return [os.path.join(csv_topics_path, csv_file) for csv_file in csv_files]

# プロンプト生成
def generate_prompt(current_utterance, previous_utterances):
    previous_utterance_pairs = "\n".join([f"AE_ID {turn.ae_id}: \"{turn.sentence}\"" for turn in previous_utterances])

    prompt = f"""Analyze the following dialogue turn and identify any influencing past turns.
Please respond with only the AE_ID of the turns that influenced the current turn. Notice that there should be only one correct answer.
Do not include any explanations or additional information.

Current turn AE_ID {current_utterance.ae_id}:
"{current_utterance.sentence}"

Past dialogue turns:
{previous_utterance_pairs}

Please answer only the ID consisting of alphabets and numbers.
Answer:
"""
    return prompt

# GPT-4oのモックメソッド
def mock_gpt4o_api(prompt):
    # Simulate the decision-making process of the model
    import random
    choices = ["AE_ID 123", "AE_ID 456", "NONE"]
    selected_choice = random.choice(choices)
    return {"result": selected_choice}

def get_chat_response(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# プロンプトの長さを保持するリスト
prompt_lengths = []

def store_prompt_length(prompt):
    length = len(prompt)
    prompt_lengths.append(length)
    return length

# コストを概算する関数
def calculate_cost(prompt_lengths):
    tokens_per_char = 0.25  # 仮のトークン数
    cost_per_token = 0.00000015
    total_tokens = sum([length * tokens_per_char for length in prompt_lengths])
    total_cost = total_tokens * cost_per_token
    return total_cost

# メイン処理
def main():
    csv_file_list = get_csv_files(csv_topics_path)

    overall_true_count = 0  # 全体のTrueのカウント
    overall_total_count = 0  # 全体のターン数

    # tqdmを使用してファイルの進行状況を表示
    for csv_file in tqdm(csv_file_list, desc="Processing CSV files"):
        tmp_turns = DialogueTurn.from_csv(csv_file)
        dialogue_turns = DialogueTurn.remove_none_relationships(tmp_turns)


        true_count = 0  # ファイルごとのTrueのカウント
        total_count = 0  # ファイルごとのターン数

        #ターンごとの結果を保持するリスト
        one_turn_result_list = []


        for index, turn in tqdm(enumerate(dialogue_turns), total=len(dialogue_turns), desc=f"Processing {os.path.basename(csv_file)}", leave=False):
            if index > 0:
                # 前の5個（またはそれ以下）のturnを取得
                start_index = max(0, index - 5)
                previous_utterances = dialogue_turns[start_index:index]

                # プロンプト生成
                prompt = generate_prompt(turn, previous_utterances)

                # GPT-4o APIに送信
                result = get_chat_response(prompt)
                judgement = DialogueTurn.relationship_exists(dialogue_turns, turn.ae_id, result)

                # True の場合カウントを増やす
                if judgement:
                    true_count += 1
                total_count += 1

                #previous_utterancesからtarget_node_id_listを取得
                target_node_id_list = [turn.ae_id for turn in previous_utterances]

                one_turn_result_list.append(OneTurnResult(index, turn.ae_id, target_node_id_list, prompt, result, turn.source, judgement))

            # rootノードの処理
            else:
                previous_utterances = []
                prompt = 'NONE'
                result = "ROOT"
                target_node_id_list = []
                judgement ='NONE'
                one_turn_result_list.append(OneTurnResult(index, turn.ae_id, target_node_id_list, prompt, result, turn.source, judgement))


        # True の割合を計算して表示
        if total_count > 0:
            true_ratio = true_count / total_count
        else:
            true_ratio = 0

        # ファイルごとの結果を保存
        result = Result(file_name=csv_file, use_model=MODEL, use_method=METHOD, rate=true_ratio,one_turn_results=one_turn_result_list)

        # ファイルごとのカウントを全体に加算
        overall_true_count += true_count
        overall_total_count += total_count

        #保存
        result.save()

        break



    # 全体の正答率を計算して表示
    if overall_total_count > 0:
        overall_true_ratio = overall_true_count / overall_total_count
    else:
        overall_true_ratio = 0
    print(f"Overall True Judgement Ratio: {overall_true_ratio:.2%}")

if __name__ == '__main__':
    main()
