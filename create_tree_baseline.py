import os
import csv
import sys
from openai import OpenAI
from tqdm import tqdm  # tqdmをインポート
from string import Template  # プレースホルダを扱うためにimport

# 自作クラスのインポート
from classes.meeting import Meeting
from classes.dialogue_turn import DialogueTurn
from classes.result_json import OneTurnResult
from classes.result_json import Result

# ファイルのパス
dir_path = os.path.dirname(os.path.abspath(__file__))
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_path = os.path.join(dir_path, 'result')
result_json_path = os.path.join(dir_path, 'result_json')
prompt_dir = os.path.join(dir_path, 'prompt')  # プロンプトファイルのディレクトリ

# 実験設定
MODEL = "gpt-4o-mini"
METHOD = "baseline"

# CSVファイルのパスを取得
def get_csv_files(csv_topics_path):
    csv_files = [f for f in os.listdir(csv_topics_path) if f.endswith('.csv')]
    return [os.path.join(csv_topics_path, csv_file) for csv_file in csv_files]

# プロンプト生成: テンプレートファイルを読み込み、プレースホルダを埋め込む
def generate_prompt_from_template(current_utterance, previous_utterances, template_file):
    # テンプレートファイルを読み込み
    with open(template_file, 'r', encoding='utf-8') as file:
        template_content = file.read()

    # string.Templateを使ってプレースホルダを埋め込む
    template = Template(template_content)

    # 前のターンのペアを作成
    previous_utterance_pairs = "\n".join([f"AE_ID {turn.ae_id} (Index: {turn.index}, Speaker: {turn.speaker}): \"{turn.sentence}\"" for turn in previous_utterances])

    # プレースホルダに値を埋め込む
    filled_prompt = template.substitute(
        current_ae_id=current_utterance.ae_id,
        current_index=current_utterance.index,
        current_speaker=current_utterance.speaker,
        current_sentence=current_utterance.sentence,
        previous_utterances=previous_utterance_pairs
    )

    return filled_prompt


def get_chat_response(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1
    )
    return completion.choices[0].message.content

# メイン処理
def main():
    csv_file_list = get_csv_files(csv_topics_path)[:10]

    overall_true_count = 0
    overall_total_count = 0

    # tqdmを使用してファイルの進行状況を表示
    for csv_file in tqdm(csv_file_list, desc="Processing CSV files"):
        tmp_turns = DialogueTurn.from_csv(csv_file)
        dialogue_turns, removed_turns = DialogueTurn.remove_none_relationships(tmp_turns)

        true_count = 0
        total_count = 0

        #ターンごとの結果を保持するリスト
        one_turn_result_list = []

        for index, turn in tqdm(enumerate(dialogue_turns), total=len(dialogue_turns), desc=f"Processing {os.path.basename(csv_file)}", leave=False):
            if index > 0:
                start_index = max(0, index - 5)
                previous_utterances = dialogue_turns[start_index:index]

                # プロンプト生成
                template = 'ver.1.0.txt'
                prompt_file = os.path.join(prompt_dir, template)
                prompt = generate_prompt_from_template(turn, previous_utterances, prompt_file)

                # GPT-4o APIに送信
                result = get_chat_response(prompt)
                judgement = DialogueTurn.relationship_exists(dialogue_turns, turn.ae_id, result)

                if judgement:
                    true_count += 1
                total_count += 1

                target_node_id_list = [turn.ae_id for turn in previous_utterances]

                one_turn_result_list.append(OneTurnResult(index, turn.ae_id, target_node_id_list, prompt, result, turn.source, judgement))

            else:
                previous_utterances = []
                prompt = 'NONE'
                result = "ROOT"
                target_node_id_list = []
                judgement = 'NONE'
                one_turn_result_list.append(OneTurnResult(index, turn.ae_id, target_node_id_list, prompt, result, turn.source, judgement))

        if total_count > 0:
            true_ratio = true_count / total_count
        else:
            true_ratio = 0

        result = Result(file_name=csv_file, use_model=MODEL, use_method=METHOD, template=template,rate=true_ratio, total_node_count=len(tmp_turns), removed_node_count=len(tmp_turns)-len(dialogue_turns), removed_node_list=removed_turns, one_turn_results=one_turn_result_list)

        overall_true_count += true_count
        overall_total_count += total_count

        result.save()


    if overall_total_count > 0:
        overall_true_ratio = overall_true_count / overall_total_count
    else:
        overall_true_ratio = 0
    print(f"Overall True Judgement Ratio: {overall_true_ratio:.2%}")

if __name__ == '__main__':
    main()
