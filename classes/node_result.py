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
    def create_csv_header(output_dir_path):

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

        #csvファイルを作成
        csv_file_path = os.path.join(output_dir_path, 'NodeResult.csv')
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

    # 書き出されたCSVファイルを読み込んで、NodeResultのリストを返す
    @staticmethod
    def load_from_csv(csv_file_path):
        node_results = []
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                node_result = NodeResult.__new__(NodeResult)  # Create an uninitialized instance
                node_result.file_name = row[0]
                node_result.turn_number = int(row[1])
                node_result.current_node_id = row[2]
                node_result.gpt_ans_node_id = row[3]
                node_result.ans_node_id = row[4]
                node_result.judgement = row[5]
                node_result.current_node_id_index = int(row[6])
                node_result.gpt_ans_node_id_index = int(row[7])
                node_result.ans_node_id_index = int(row[8])
                node_result.distance_from_ans = int(row[9])
                node_result.contains_answer = row[10]
                node_results.append(node_result)
        return node_results

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
