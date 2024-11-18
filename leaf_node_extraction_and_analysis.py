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

                if len(sentence.split()) <= 6:
                    leaf_nodes.append(sentence)

        return leaf_nodes

def test(sentence):
    sentence = sentence.replace(',', '').replace('.', '').replace('?', '').replace('!', '')

    return len(sentence.split())


def main():
    # CSVファイルのパスを取得
    csv_files = get_csv_files(csv_topics_path)

    # 単語とその出現回数を格納する辞書
    word_count = {}

    for csv_file in csv_files:
        leaf_nodes = extract_short_leaf_nodes(csv_file)
        for leaf_node in leaf_nodes:
            words = leaf_node.split()
            for word in words:
                w = word.lower()
                if w in word_count:
                    word_count[w] += 1
                else:
                    word_count[w] = 1

    # 出現回数が多い順にソート
    word_count = dict(sorted(word_count.items(), key=lambda x: x[1], reverse=True))

    # 出現回数が多い 上位10単語を表示
    print(list(word_count.items())[:30])

    # print(test('Yeah, got the email.'))

if __name__ == '__main__':
    main()
