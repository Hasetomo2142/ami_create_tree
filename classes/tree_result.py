import csv
import json
import os
import re
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 自作クラスのインポート
from .dialogue_turn import DialogueTurn
from .result_json import Result

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
csv_topics_path = os.path.join(dir_path, 'CSV_topics')
result_json_path = os.path.join(dir_path, 'result_json')

class TreeResult:
    def __init__(self, Result):

        self.estimated_tree = nx.DiGraph()
        self.real_tree = nx.DiGraph()

        self.build_estimated_tree(Result)
        self.build_real_tree(Result)

        match = re.search(r'-\s*([^\.]+)\.', Result.file_name)
        self.file_name = match.group(1).strip()
        self.rate = Result.rate
        self.num_solved_problems = Result.total_node_count - Result.removed_node_count
        self.num_options_without_answer = self.get_num_options_without_answer(Result)
        self.build_tree_depth = self.get_tree_depth(self.estimated_tree)
        self.real_tree_depth = self.get_tree_depth(self.real_tree)
        self.tree_edit_distance = nx.graph_edit_distance(self.estimated_tree, self.real_tree)
        self.edge_similarity = self.edge_similarity()
        self.path_based_similarity = self.compute_path_based_similarity()

        # print(f"Estimated Tree - Nodes: {self.estimated_tree.number_of_nodes()}, Edges: {self.estimated_tree.number_of_edges()}")
        # print(f"Real Tree - Nodes: {self.real_tree.number_of_nodes()}, Edges: {self.real_tree.number_of_edges()}")



    def get_num_options_without_answer(self, Result):
        count = 0
        for turn in Result.one_turn_results:
            if not turn.contains_answer:
                count += 1
        return count


    def build_estimated_tree(self, Result):
        one_turn_results = Result.one_turn_results

        for one_turn_result in one_turn_results:
            current_node_ae_id = one_turn_result.current_node.ae_id

            # 現在のノードがすでに存在しているか確認（ae_id で照合）
            if not self.estimated_tree.has_node(current_node_ae_id):
                self.estimated_tree.add_node(current_node_ae_id)

            # GPTの回答ノードを取得して、エッジを追加（ae_id で処理）
            if one_turn_result.current_node.source != 'NONE':
                gpt_ans_node = DialogueTurn.find_by_ae_id(one_turn_result.gpt_ans)

                if gpt_ans_node is not None:
                    gpt_ans_node_ae_id = gpt_ans_node.ae_id

                    # ノードがすでに追加されているか確認（ae_id で確認）
                    if not self.estimated_tree.has_node(gpt_ans_node_ae_id):
                        self.estimated_tree.add_node(gpt_ans_node_ae_id)

                    # エッジを追加（ae_id を用いる）
                    self.estimated_tree.add_edge(gpt_ans_node_ae_id, current_node_ae_id)

        # グラフを描画し保存
        nx.draw(self.estimated_tree, with_labels=False)
        plt.savefig('estimated.png')
        plt.close()


    def build_real_tree(self, Result):
        one_turn_results = Result.one_turn_results

        for one_turn_result in one_turn_results:
            current_node_ae_id = one_turn_result.current_node.ae_id

            # 現在のノードがすでに存在しているか確認（ae_id で照合）
            if not self.real_tree.has_node(current_node_ae_id):
                self.real_tree.add_node(current_node_ae_id)

            # 答えノードを取得し、エッジを追加（ae_id で処理）
            if one_turn_result.current_node.source != 'NONE':
                ans_node = DialogueTurn.find_by_ae_id(one_turn_result.ans)

                if ans_node is not None:
                    ans_node_ae_id = ans_node.ae_id

                    # ノードがすでに追加されているか確認（ae_id で確認）
                    if not self.real_tree.has_node(ans_node_ae_id):
                        self.real_tree.add_node(ans_node_ae_id)

                    # エッジを追加（ae_id を用いる）
                    self.real_tree.add_edge(ans_node_ae_id, current_node_ae_id)

        # グラフを描画し保存
        nx.draw(self.real_tree, with_labels=False)
        plt.savefig('real.png')
        plt.close()

    def edge_similarity(self):
        """
        2つのグラフ間のエッジのコサイン類似度を計算するメソッド。
        :return: コサイン類似度 (0から1の範囲)
        """
        # 1. 両方のグラフの全エッジのリストを作成（エッジの和集合）
        edges_estimated = set(self.estimated_tree.edges())
        edges_real = set(self.real_tree.edges())
        all_edges = list(edges_estimated.union(edges_real))

        # 2. 各グラフにおけるエッジの存在を0と1で表すベクトルを作成
        vec_estimated = np.array([1 if edge in edges_estimated else 0 for edge in all_edges])
        vec_real = np.array([1 if edge in edges_real else 0 for edge in all_edges])

        # 3. コサイン類似度の計算
        cosine_sim = cosine_similarity([vec_estimated], [vec_real])

        # 4. コサイン類似度を返す
        return cosine_sim[0][0]

    def get_tree_depth(self, tree):
        # 木が持つ全てのノードを取得
        nodes = list(tree.nodes())

        # ルートノードを取得
        root_node = None
        for node in nodes:
            # もしノードが 'NONE' である場合を想定して、ルートノードを見つける
            if DialogueTurn.find_by_ae_id(node).source == 'NONE':
                root_node = node
                break

        # ルートノードが見つからなかった場合のエラー処理
        if root_node is None:
            raise ValueError("No root node found in the tree")

        # ルートノードからの最大深さを計算
        path_lengths = nx.single_source_shortest_path_length(tree, root_node)

        # 最長のパスの長さ（深さ）を返す
        max_depth = max(path_lengths.values())

        return max_depth



    def compute_path_based_similarity(self):
        """
        最短経路に基づくグラフの類似度を計算するメソッド。
        :return: コサイン類似度 (0から1の範囲)
        """
        def compute_shortest_paths(graph):
            # ノードがオブジェクトかどうかを確認して適切にソート
            nodes = graph.nodes()
            if isinstance(next(iter(nodes)), str) or isinstance(next(iter(nodes)), int):
                # ノードが文字列または数値の場合、そのままソート
                sorted_nodes = sorted(nodes)
            else:
                # ノードがオブジェクトで、ae_id を持つ場合、その属性でソート
                sorted_nodes = sorted(nodes, key=lambda node: getattr(node, 'ae_id', node))

            shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
            path_lengths = []

            for i, node1 in enumerate(sorted_nodes):
                for node2 in sorted_nodes[i+1:]:
                    if node2 in shortest_path_lengths[node1]:
                        path_lengths.append(shortest_path_lengths[node1][node2])
                    else:
                        # 経路が存在しない場合は非常に大きな値を追加
                        path_lengths.append(1e6)  # 大きな値に置き換え
            return np.array(path_lengths)

        # 推定ツリーと実際のツリーの最短経路を計算
        paths_estimated = compute_shortest_paths(self.estimated_tree)
        paths_real = compute_shortest_paths(self.real_tree)

        # 最小の長さに基づいてベクトルを切り詰める
        min_length = min(len(paths_estimated), len(paths_real))
        paths_estimated = paths_estimated[:min_length]
        paths_real = paths_real[:min_length]

        # コサイン類似度を計算（無限大の値を大きな数に置き換えたのでエラーが発生しない）
        cosine_sim = cosine_similarity([paths_estimated], [paths_real])
        return cosine_sim[0][0]



    # CSVファイルのヘッダーを作成してCSVファイルに書き込む
    @staticmethod
    def create_csv_header(use_method, template):
        """
        CSVファイルのヘッダーを作成するメソッド。
        :return: ヘッダーのリスト
        """
        header = [
            'file_name',
            'rate',
            'num_solved_problems',
            'num_options_without_answer',
            'build_tree_depth',
            'real_tree_depth',
            'tree_edit_distance',
            'edge_similarity',
            'path_based_similarity'
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
        csv_file_path = os.path.join(prompt_dir_path, 'TreeResult.csv')
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        return csv_file_path

    # Result単位でCSVファイルにレコードを追加
    def save_one_tree(self, csv_file_path):
        """
        1つのTreeResultオブジェクトをCSVファイルに書き込むメソッド。
        :param csv_file_path: CSVファイルのパス
        """
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.file_name,
                self.rate,
                self.num_solved_problems,
                self.num_options_without_answer,
                self.build_tree_depth,
                self.real_tree_depth,
                self.tree_edit_distance,
                self.edge_similarity,
                self.path_based_similarity
            ])

    @staticmethod
    def save_trees_from_result_class(result, tree_result_csv_path):
        tree_result = TreeResult(result)
        tree_result.save_one_tree(tree_result_csv_path)
