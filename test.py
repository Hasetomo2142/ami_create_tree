import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dialogue_turn import DialogueTurn
from classes.result_json import Result
from classes.node_result import NodeResult

# 指定されたCSVファイルを1行ずつ読み込む。1行目はヘッダーとして読み飛ばす
def read_csv_file(file_path):
    """指定されたCSVファイルを1行ずつ読み込む。1行目はヘッダーとして読み飛ばす"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーを読み飛ばす
        for row in reader:
            yield row

# 全ノード数と回答が含まれていないノード数をカウントする
def count_node_results(file_path):
    """CSVファイルを読み込んで全ノード数と回答が含まれていないノード数をカウントする"""
    all_node_count = 0
    not_contains_answer_count = 0

    for row in read_csv_file(file_path):
        if row[3] != 'ROOT':
            all_node_count += 1
            if row[10] == 'False':
                not_contains_answer_count += 1

    return all_node_count, not_contains_answer_count

# 外れ値を除去する
def remove_outliers(data, m=2):
    """与えられたリストの外れ値を除去する。デフォルトは標準偏差2倍以上の値を除外"""
    mean = sum(data) / len(data)
    return [x for x in data if abs(x - mean) < m * (sum((xi - mean) ** 2 for xi in data) / len(data)) ** 0.5]

# ノードインデックスの分布を描画する
def plot_node_index_distribution(file_path, output_image_path):
    """CSVファイルからノードインデックスの分布を描画し、画像として保存する"""
    node_index_list = []

    for row in read_csv_file(file_path):
        node_index_list.append(int(row[9]))

    # 外れ値を除去
    node_index_list = remove_outliers(node_index_list)

    # ヒストグラムを描画して保存
    sns.histplot(node_index_list, bins=range(min(node_index_list), max(node_index_list) + 2, 1))
    plt.xlabel('Distance')  # 横軸にラベルを追加
    plt.savefig(output_image_path)
    plt.close()

# メイン処理
if __name__ == '__main__':
    csv_file_path = '/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.1-20.txt/NodeResult.csv'
    output_image_path = '/home/hasegawa_tomokazu/create_tree/node_index_distribution.png'

    # 全ノード数と回答が含まれていないノード数のカウント
    all_node_count, not_contains_answer_count = count_node_results(csv_file_path)
    print(f"All nodes: {all_node_count}")
    print(f"Nodes without answer: {not_contains_answer_count}")

    print(f"{(all_node_count - not_contains_answer_count) / all_node_count:.1%}")

    # ノードインデックスの分布を画像に保存
    plot_node_index_distribution(csv_file_path, output_image_path)
    print(f"Node index distribution plot saved to {output_image_path}")
