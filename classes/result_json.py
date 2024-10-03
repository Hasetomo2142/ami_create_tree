import csv
import json
import os
import re
import datetime

# 自作クラスのインポート
from .dialogue_turn import DialogueTurn

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_json_path = os.path.join(dir_path, 'result_json')

class OneTurnResult:
	def __init__(self, turn_number, current_node_id, target_node_id_list, prompt, gpt_ans, ans, judgement):
		self.turn_number = turn_number
		self.current_node = DialogueTurn.find_by_ae_id(current_node_id)
		self.target_node_list = [DialogueTurn.find_by_ae_id(target_node_id) for target_node_id in target_node_id_list]
		self.prompt = prompt
		self.gpt_ans = gpt_ans
		self.ans = ans
		self.judgement = judgement

		# 正解が含まれているか (True/False)　カラムとしては保持しない
		self.contains_ans = self.ans in self.target_node_list

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

	@classmethod
	def from_dict(cls, data):
		current_node = DialogueTurn.from_dict(data["current_node"])
		target_node_list = [DialogueTurn.from_dict(node_data) for node_data in data["target_node_list"]]
		return cls(
				turn_number=data["turn_number"],
				current_node=current_node,
				target_node_list=target_node_list,
				prompt=data["prompt"],
				gpt_ans=data["gpt_ans"],
				ans=data["ans"],
				judgement=data["judgement"]
		)

	def contains_answer(self):
		for target_node in self.target_node_list:
			if target_node.ae_id == self.ans:
				return True
		return False

class Result:
	def __init__(self, file_name, use_model, use_method, template, rate, total_node_count, removed_node_count, removed_node_list, one_turn_results):
		self.file_name = file_name
		self.use_model = use_model
		self.use_method = use_method
		self.template = template
		self.rate = rate
		self.total_node_count =total_node_count
		self.removed_node_count = removed_node_count
		self.removed_node_list = removed_node_list
		self.one_turn_results = one_turn_results

	def soleve_count(self):
		return self.total_node_count - self.removed_node_count - 1 # -1はROOTノード

	def true_count(self):
		true_count = 0
		for one_turn_result in self.one_turn_results:
			if one_turn_result.judgement == True:
				true_count += 1
		return true_count

	def calc_accuracy(self):
		return self.true_count / self.soleve_count

	def to_dict(self):
		return {
			"file_name": self.file_name,
			"use_model": self.use_model,
			"use_method": self.use_method,
			"template": self.template,
			"rate": self.rate,
			"total_node_count": self.total_node_count,
			"removed_node_count": self.removed_node_count,
			"removed_node_list": [removed_node.to_dict() for removed_node in self.removed_node_list],
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

		# use_methodのディレクトリにpromptのディレクトリを作成
		prompt_dir_path = os.path.join(method_dir_path, self.template)
		if not os.path.exists(prompt_dir_path):
			os.makedirs(prompt_dir_path)

		# jsonを保存
		result_json_file = os.path.join(prompt_dir_path, os.path.basename(self.file_name).replace('.csv', '.json'))
		with open(result_json_file, 'w') as f:
			f.write(self.to_json())

	@classmethod
	def from_dict(cls, data):
		one_turn_results = []
		for one_turn_result_data in data["one_turn_results"]:

			target_node_list = []
			for target_node in one_turn_result_data['target_node_list']:
				target_node_list.append(target_node['ae_id'])

			one_turn_result = OneTurnResult(
					one_turn_result_data['turn_number'],
					one_turn_result_data['current_node']["ae_id"],
					target_node_list,
					one_turn_result_data['prompt'],
					one_turn_result_data['gpt_ans'],
					one_turn_result_data['ans'],
					one_turn_result_data['judgement']
					)
			one_turn_results.append(one_turn_result)

		removed_node_list = []
		for removed_node_data in data["removed_node_list"]:
			removed_node = DialogueTurn.from_dict(removed_node_data)
			removed_node_list.append(removed_node)
		return cls(
				file_name=data["file_name"],
				use_model=data["use_model"],
				use_method=data["use_method"],
				template=data["template"],
				rate=data["rate"],
				total_node_count=data["total_node_count"],
				removed_node_count=data["removed_node_count"],
				removed_node_list=removed_node_list,
				one_turn_results=one_turn_results
		)

	def load_result_from_json(json_file_path):
		with open(json_file_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		result = Result.from_dict(data)
		return result
