import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dialogue_turn import DialogueTurn
from classes.result_json import Result
from classes.node_result import NodeResult

node_results_1 = NodeResult.load_from_csv('/home/hasegawa_tomokazu/create_tree/result_json/baseline_no_reindex/ver.1.0.txt/NodeResult.csv')
node_results_5 = NodeResult.load_from_csv('/home/hasegawa_tomokazu/create_tree/result_json/baseline_no_reindex/ver.1.1-5.txt/NodeResult.csv')

# ファイルのパス
dir_path = os.path.dirname(os.path.abspath(__file__))
csv_topics_path = os.path.join(dir_path, 'CSV_topics')

count = 0
parent_distance_1 = 0


for index, (node_result_1, node_result_5) in enumerate(zip(node_results_1, node_results_5)):
    # csv_topics_pathの中からnode_result_1.file_nameを含むファイルを検索
    for file_name in os.listdir(csv_topics_path):
        if node_result_1.file_name in file_name:
            file_path = os.path.join(csv_topics_path, file_name)
            break
    else:
        raise ValueError(f"File not found for node_result_1.file_name: {node_result_1.file_name}")

    tmp_turns = DialogueTurn.from_csv(file_path)
    dialogue_turns, removed_turns = DialogueTurn.remove_none_relationships(tmp_turns)

    reindex_turns = DialogueTurn.reindex(dialogue_turns)
    ae_id_to_index = {turn.ae_id: turn.index for turn in reindex_turns}

    if node_result_1.turn_number != 0 and  node_result_1.ans_node_id != 'NONE':
        try:
            current_node_index = ae_id_to_index[node_result_1.current_node_id]
            parent_node_index = ae_id_to_index[node_result_1.ans_node_id]
            count += 1

            if current_node_index - parent_node_index == 1:
                parent_distance_1 += 1


        except KeyError as e:
            print(f"KeyError: {e} not found in ae_id_to_index. Skipping this node result.")
            continue

print('count',count)
print('parent_distance_1',parent_distance_1)
