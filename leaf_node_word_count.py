import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dialogue_turn import DialogueTurn
from classes.result_json import Result
from classes.node_result import NodeResult
import networkx as nx

csv_topics_path = '/home/hasegawa_tomokazu/create_tree/CSV_topics'

# ファイルのパスを取得
def get_csv_files(csv_topics_path):
    csv_files = [f for f in os.listdir(csv_topics_path) if f.endswith('.csv')]
    return [os.path.join(csv_topics_path, csv_file) for csv_file in csv_files]

def extract_short_leaf_nodes(csv_file):
    leaf_nodes = []

    # CSVファイルを読み込む
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)

        # ヘッダーをスキップ
        next(reader)

        # ルートノードはスキップ
        next(reader)

        for row in reader:
            if row[5] != 'NONE' and row[6] == 'NONE':
                # row[4]から句読点を削除
                sentence = row[4].replace(',', '').replace('.', '').replace('?', '').replace('!', '')

                # if len(sentence.split()) <= 6:
                leaf_nodes.append(sentence)

        return leaf_nodes

def test(sentence):
    sentence = sentence.replace(',', '').replace('.', '').replace('?', '').replace('!', '')

    return len(sentence.split())


def main():
    # CSVファイルのパスを取得
    csv_files = get_csv_files(csv_topics_path)

    length_size_count = []

    for csv_file in csv_files:
        leaf_nodes = extract_short_leaf_nodes(csv_file)
        for leaf_node in leaf_nodes:
            word_count = len(leaf_node.split())
            length_size_count.append(word_count)

    # 外れ値を削除
    length_size_count = [l for l in length_size_count if l <= 30]

    # ヒストグラムを描画
    sns.histplot(length_size_count, bins=range(min(length_size_count), max(length_size_count) + 1))
    plt.xlabel('Number of words')
    plt.ylabel('Number of leaf nodes')
    plt.title('Number of words in leaf nodes')
    # pngファイルとして保存
    plt.savefig('word_count.png')


if __name__ == '__main__':
    main()
