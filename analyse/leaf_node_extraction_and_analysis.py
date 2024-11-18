import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dialogue_turn import DialogueTurn
from classes.result_json import Result
from classes.node_result import NodeResult

node_results_1 = NodeResult.load_from_csv('/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.0.txt/NodeResult.csv')
node_results_5 = NodeResult.load_from_csv('/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.1.txt/NodeResult.csv')

# ファイルのパス
dir_path = os.path.dirname(os.path.abspath(__file__))
csv_topics_path = os.path.join(dir_path, 'CSV_topics')

