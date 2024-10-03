import csv
import json
import os
import re
import datetime

# 自作クラスのインポート
from .dialogue_turn import DialogueTurn
from .result_json import Result

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_json_path = os.path.join(dir_path, 'result_json')

class NodeResult:
    def __init__(self, OneTurnResult):
        self.file_name = ''  # ファイル名は後から取得
        self.turn_number = OneTurnResult.turn_number
        self.current_node_id = OneTurnResult.current_node.ae_id
        self.gpt_ans_node_id = OneTurnResult.gpt_ans
        self.ans_node_id = OneTurnResult.ans
        self.judgement = OneTurnResult.judgement

        # current_node_id_indexの取得
        current_result = DialogueTurn.find_by_ae_id(self.current_node_id)

        if current_result == "NONE":
            self.current_node_id_index = 0
        elif hasattr(current_result, 'index'):
            self.current_node_id_index = current_result.index
        else:
            raise ValueError(f"Unexpected result from find_by_ae_id for current_node_id: {self.current_node_id}, result: {current_result}")

        # gpt_ans_node_id_indexの取得
        gpt_ans_result = DialogueTurn.find_by_ae_id(self.gpt_ans_node_id)

        if gpt_ans_result == "NONE":
            self.gpt_ans_node_id_index = 0
        elif hasattr(gpt_ans_result, 'index'):
            self.gpt_ans_node_id_index = gpt_ans_result.index
        else:
            raise ValueError(f"Unexpected result from find_by_ae_id for gpt_ans_node_id: {self.gpt_ans_node_id}, result: {gpt_ans_result}")

        # ans_node_id_indexの取得
        ans_result = DialogueTurn.find_by_ae_id(self.ans_node_id)

        if ans_result == "NONE":
            self.ans_node_id_index = 0
        elif hasattr(ans_result, 'index'):
            self.ans_node_id_index = ans_result.index
        else:
            raise ValueError(f"Unexpected result from find_by_ae_id for ans_node_id: {self.ans_node_id}, result: {ans_result}")


        # 正常に数値を取得した後で、distance_from_ans を計算
        self.distance_from_ans = abs(self.current_node_id_index - self.ans_node_id_index)

        # contains_answerの処理 (関数でない場合は括弧を外す)
        self.contains_answer = OneTurnResult.contains_answer()


    # CSVファイルのヘッダーを作成してCSVファイルに書き込む
    @staticmethod
    def create_csv_header(use_method, template):

        header = [
            'file_name',
            'turn_number',
            'current_node_id',
            'gpt_ans_node_id',
            'ans_node_id',
            'judgement',
            'current_node_id_index',
            'gpt_ans_node_id_index',
            'ans_node_id_index',
            'distance_from_ans',
            'contains_answer'
        ]

        # result_json_pathにuse_methodのディレクトリを作成
        method_dir_path = os.path.join(result_json_path, use_method)
        if not os.path.exists(method_dir_path):
            os.makedirs(method_dir_path)

        # use_methodのディレクトリにpromptのディレクトリを作成
        prompt_dir_path = os.path.join(method_dir_path, template)
        if not os.path.exists(prompt_dir_path):
            os.makedirs(prompt_dir_path)

        #csvファイルを作成
        csv_file_path = os.path.join(prompt_dir_path, 'NodeResult.csv')
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        return csv_file_path

    # Result単位でCSVファイルにレコードを追加
    def save_one_node(self, csv_file_path):
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.file_name,
                self.turn_number,
                self.current_node_id,
                self.gpt_ans_node_id,
                self.ans_node_id,
                self.judgement,
                self.current_node_id_index,
                self.gpt_ans_node_id_index,
                self.ans_node_id_index,
                self.distance_from_ans,
                self.contains_answer
            ])
            
    @staticmethod
    def save_nodes_from_result_class(result, csv_file_path):
        node_results = []
        for one_turn_result in result.one_turn_results:
            node_result = NodeResult(one_turn_result)
            match = re.search(r'-\s*([^\.]+)\.', result.file_name)
            node_result.file_name = match.group(1).strip()
            node_results.append(node_result)

        for node_result in node_results:
            node_result.save_one_node(csv_file_path)
