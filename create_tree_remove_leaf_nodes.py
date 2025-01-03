import json
import os
import csv
import sys
import time
from openai import OpenAI
from tqdm import tqdm  # tqdmをインポート
from string import Template  # プレースホルダを扱うためにimport

# 自作クラスのインポート
from classes.meeting import Meeting
from classes.dialogue_turn import DialogueTurn
from classes.result_json import OneTurnResult
from classes.result_json import Result
from classes.node_result import NodeResult
from classes.tree_result import TreeResult
from classes.gpt_cost_calculator import GPTCostCalculator

# ファイルのパス
dir_path = os.path.dirname(os.path.abspath(__file__))
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_path = os.path.join(dir_path, 'result')
result_json_path = os.path.join(dir_path, 'result_json')
prompt_dir = os.path.join(dir_path, 'prompt')  # プロンプトファイルのディレクトリ

# 実験設定
MODEL = "gpt-4o-mini"
METHOD = "baseline"
PROMPT = "ver.1.1.txt"
COUNT = 50

# 参照する直近ノードの数
RECENT_TURNS = 5


def get_output_dir_path():
    # result_json_pathにuse_methodのディレクトリを作成
    method_dir_path = os.path.join(result_json_path, METHOD)
    if not os.path.exists(method_dir_path):
        os.makedirs(method_dir_path)

    # use_methodのディレクトリにpromptのディレクトリを作成
    prompt_dir_path = os.path.join(method_dir_path, PROMPT)
    if not os.path.exists(prompt_dir_path):
        os.makedirs(prompt_dir_path)

    return prompt_dir_path

#結果を格納するファイルを生成
out_put_dir_path = get_output_dir_path()
node_result_csv_path = NodeResult.create_csv_header(out_put_dir_path)
tree_result_csv_path = TreeResult.create_csv_header(out_put_dir_path)

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
        temperature=0.0,
    )
    return completion.choices[0].message.content

# メイン処理
def main():
    print(f"Model: {MODEL}")
    print(f"Method: {METHOD}")
    print(f"Prompt: {PROMPT}")
    print(f"Count: {COUNT}")

    csv_file_list = get_csv_files(csv_topics_path)[:COUNT]
    # csv_file_list = get_csv_files(csv_topics_path)

    overall_true_count = 0
    overall_total_count = 0

    # 処理前の時刻を記録
    start_time = time.time()

    # APIのコスト計算クラスをインスタンス化
    gpt_cout_calculator = GPTCostCalculator(MODEL)

    # tqdmを使用してファイルの進行状況を表示
    for csv_file in tqdm(csv_file_list, desc="Processing CSV files"):
        result_exists = False
        result_file = os.path.join(out_put_dir_path, f"{os.path.basename(csv_file).replace('.csv', '')}.json")
        if os.path.exists(result_file):
            result_exists = True

        try:
            tmp_turns = DialogueTurn.from_csv(csv_file)
            dialogue_turns, removed_turns = DialogueTurn.remove_none_relationships(tmp_turns)

            true_count = 0
            total_count = 0

            #ターンごとの結果を保持するリスト
            one_turn_result_list = []

            # ae_idとindexの対応を保持する辞書
            ae_id_to_index = {turn.ae_id: turn.index for turn in dialogue_turns}

            for index, turn in tqdm(enumerate(dialogue_turns), total=len(dialogue_turns), desc=f"Processing {os.path.basename(csv_file)}", leave=False):
                # ルートノードの処理
                if index == 0:
                    previous_utterances = []
                    prompt = 'NONE'
                    result = "ROOT"
                    target_node_id_list = []
                    judgement = 'NONE'
                    one_turn_result_list.append(OneTurnResult(index, turn.ae_id, target_node_id_list, prompt, result, turn.source, judgement, ae_id_to_index))

                else:
                    start_index = max(0, index - RECENT_TURNS)
                    previous_utterances = dialogue_turns[start_index:index]

                    #　previous_utterancesから葉ノードを除外　要素に対してis_leafメソッドを適用しtrueのものを除外
                    previous_utterances = [turn for turn in previous_utterances if not turn.is_leaf()]


                    # プロンプト生成
                    template = PROMPT
                    prompt_file = os.path.join(prompt_dir, template)
                    prompt = generate_prompt_from_template(turn, previous_utterances, prompt_file)
                    gpt_cout_calculator.add_input_text(prompt)

                    # out_put_dir_pathにすでに結果がある場合は、その結果を読み込む
                    if result_exists:
                        rusult_json = Result.load_result_from_json(result_file, ae_id_to_index)
                        one_turn_results = rusult_json.one_turn_results
                        for one_turn_result in one_turn_results:
                            if one_turn_result.current_node.ae_id == turn.ae_id:
                                result = one_turn_result.gpt_ans
                                # print(f"\033[91mLoaded result from {result_file}\033[0m")
                    else:
                        # GPT-4o APIに送信
                        result = get_chat_response(prompt)

                    gpt_cout_calculator.add_output_text(result)
                    judgement = DialogueTurn.relationship_exists(dialogue_turns, turn.ae_id, result)

                    if judgement:
                        true_count += 1
                    total_count += 1

                    target_node_id_list = [turn.ae_id for turn in previous_utterances]

                    one_turn_result_list.append(
                        OneTurnResult(
                            turn_number = index,
                            current_node_id = turn.ae_id,
                            target_node_id_list = target_node_id_list,
                            prompt = prompt,
                            gpt_ans = result,
                            ans = turn.source,
                            judgement = judgement,
                            ae_id_to_index = ae_id_to_index
                            ).reindex()
                        )
        except Exception as e:
            print(csv_file)
            print(f"Error: {e}")
            continue

        if total_count > 0:
            true_ratio = true_count / total_count
        else:
            true_ratio = 0

        # 処理後の時刻を記録
        end_time = time.time()

        result = Result(
            file_name=csv_file,
            use_model=MODEL,
            use_method=METHOD,
            template=template,
            rate=true_ratio,
            total_node_count=len(tmp_turns),
            removed_node_count=len(removed_turns),
            removed_node_list=removed_turns,
            one_turn_results=one_turn_result_list
            )

        overall_true_count += true_count
        overall_total_count += total_count
        try:
            result.save()
            NodeResult.save_nodes_from_result_class(result, node_result_csv_path)
            tree_result = TreeResult.save_trees_from_result_class(result, tree_result_csv_path)
            if not result_exists:
                tree_result.draw_tree(out_put_dir_path, "estimated")
                tree_result.draw_tree(out_put_dir_path, "real")
        except Exception as e:
            print(f"Error: {e}")
            continue

    # 処理にかかった時間を計算
    elapsed_time = end_time - start_time

    # コストを計算
    cost = gpt_cout_calculator.calculate_cost()

    #正解率を計算
    if overall_total_count > 0:
        overall_true_ratio = overall_true_count / overall_total_count
    else:
        overall_true_ratio = 0

    print(f"Overall True Judgement Ratio: {overall_true_ratio:.2%}")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"Total cost: ${cost:.5f}")

    #out_put_dir_pathにもOverall True Judgement Ratioを保存
    with open(os.path.join(out_put_dir_path, 'Overall_True_Judgement_Ratio.txt'), 'w') as f:
        f.write(f"Model: {MODEL}\n")
        f.write(f"Method: {METHOD}\n")
        f.write(f"Prompt: {PROMPT}\n")
        f.write(f"Count: {COUNT}\n")
        f.write(f"Overall True Judgement Ratio: {overall_true_ratio:.2%}\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total cost: ${cost:.5f}\n")
        f.write(f"Recent Turns: {RECENT_TURNS}\n")



if __name__ == '__main__':
    main()
