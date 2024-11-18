import csv
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

become_false= 0
become_true= 0

total_count = 0
corect_count_1 = 0
corect_count_5 = 0

for index, (node_result_1, node_result_5) in enumerate(zip(node_results_1, node_results_5)):
    if node_result_1.turn_number not in [0, 1]:
        if node_result_1.judgement == "True" and node_result_5.judgement == "False":
            become_false += 1

        if node_result_1.judgement == "False" and node_result_5.judgement == "True":
            become_true += 1

            text = DialogueTurn.find_by_ae_id(node_result_5.current_node_id).sentence
            index = DialogueTurn.find_by_ae_id(node_result_5.current_node_id).index
            print(node_result_5.file_name)
            print(node_result_5.current_node_id)
            print(index+1)
            print(text)
            print('--------------------------------------------')
            #ここでenterを押すと次のデータが表示される
            # input()

        total_count += 1

        if node_result_1.judgement == "True":
            corect_count_1 += 1
        if node_result_5.judgement == "True":
            corect_count_5 += 1



print ("become_false",become_false)
print ("become_true",become_true)

print('accuracy_1:', round((corect_count_1 / total_count) * 100, 2), '%')
print('accuracy_5:', round((corect_count_5 / total_count) * 100, 2), '%')
