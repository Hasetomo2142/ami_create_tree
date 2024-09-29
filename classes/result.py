import csv
import json
import os
import re
import datetime

# 自作クラスのインポート
from classes.dialogue_turn import DialogueTurn

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_json_path = os.path.join(dir_path, 'result_json')

class OneTurnResult:
	def __init__(self, turn_number, current_node_id, target_node_id_list, prompt, gpt_ans, ans, judgement):
		self.turn_number = turn_number
		self.current_node = DialogueTurn.find_by_ae_id(csv_topics_path, current_node_id)
		self.target_node_list = [DialogueTurn.find_by_ae_id(csv_topics_path, target_node_id) for target_node_id in target_node_id_list]
		self.prompt = prompt
		self.gpt_ans = gpt_ans
		self.ans = ans
		self.judgement = judgement

	def to_dict(self):
		return {
			"turn_number": self.turn_number,
			"current_node": self.current_node.to_dict(),
			"target_node_list": [target_node.to_dict() for target_node in self.target_node_list],
			"prompt": self.prompt,
			"gpt_ans": self.gpt_ans,
			"ans": self.ans,
			"judgement": self.judgement
		}

class Result:
	def __init__(self, file_name, use_model, use_method, rate, one_turn_results):
		self.file_name = file_name
		self.use_model = use_model
		self.use_method = use_method
		self.rate = rate
		self.one_turn_results = one_turn_results

	def to_dict(self):
		return {
			"file_name": self.file_name,
			"use_model": self.use_model,
			"use_method": self.use_method,
			"rate": self.rate,
			"one_turn_results": [one_turn_result.to_dict() for one_turn_result in self.one_turn_results]
	}

	def to_json(self):
		return json.dumps(self.to_dict(), indent=4)

	#日付と使用した手法のディレクトリを作成
	def save(self):
		# result_json_pathにuse_methodのディレクトリを作成
		method_dir_path = os.path.join(result_json_path, self.use_method)
		if not os.path.exists(method_dir_path):
			os.makedirs(method_dir_path)

		# jsonを保存
		result_json_file = os.path.join(method_dir_path, os.path.basename(self.file_name).replace('.csv', '.json'))
		with open(result_json_file, 'w') as f:
			f.write(self.to_json())
