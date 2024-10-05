import csv
import json
import os
import re
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textwrap  # 改行処理のために使用

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

        self.color_map = self.get_color_map(Result.get_speakers())

        self.build_estimated_tree(Result)
        self.build_real_tree(Result)

        match = re.search(r'-\s*([^\.]+)\.', Result.file_name)
        self.file_name = match.group(1).strip()
        self.rate = Result.rate
        self.num_solved_problems = Result.total_node_count - Result.removed_node_count
        self.num_options_without_answer = self.get_num_options_without_answer(Result)


        self.build_tree_depth = self.get_tree_depth(self.estimated_tree)
        self.real_tree_depth = self.get_tree_depth(self.real_tree)

        # self.tree_edit_distance = nx.graph_edit_distance(self.estimated_tree, self.real_tree, upper_bound=50)
        # self.edge_similarity = self.edge_similarity()
        # self.path_based_similarity = self.compute_path_based_similarity()


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
            current_node = one_turn_result.current_node

            # 現在のノードがすでに存在しているか確認（ae_id で照合）
            if not self.estimated_tree.has_node(current_node.ae_id):

                node_label = f"{current_node.index + 1}: {self.wrap_and_truncate_text(current_node.sentence)}"
                self.estimated_tree.add_node(current_node.ae_id, label=node_label, color=self.color_map[DialogueTurn.find_by_ae_id(current_node.ae_id).speaker])

            # GPTの回答ノードを取得して、エッジを追加（ae_id で処理）
            if one_turn_result.current_node.source != 'NONE':
                gpt_ans_node = DialogueTurn.find_by_ae_id(one_turn_result.gpt_ans)

                if gpt_ans_node != "NONE":

                    # ノードがすでに追加されているか確認（ae_id で確認）
                    if not self.estimated_tree.has_node(gpt_ans_node.ae_id):
                        node_label = f"{gpt_ans_node.index + 1}: {self.wrap_and_truncate_text(gpt_ans_node.sentence)}"
                        self.estimated_tree.add_node(gpt_ans_node.ae_id, label=node_label, color=self.color_map[DialogueTurn.find_by_ae_id(gpt_ans_node.ae_id).speaker])

                    # エッジを追加（ae_id を用いる）
                    self.estimated_tree.add_edge(gpt_ans_node.ae_id, current_node.ae_id)

    def build_real_tree(self, Result):
        one_turn_results = Result.one_turn_results
        for one_turn_result in one_turn_results:
            current_node = one_turn_result.current_node

            if not self.real_tree.has_node(current_node.ae_id):

                node_label = f"{current_node.index + 1}: {self.wrap_and_truncate_text(current_node.sentence)}"
                self.real_tree.add_node(current_node.ae_id, label=node_label, color=self.color_map[DialogueTurn.find_by_ae_id(current_node.ae_id).speaker])

            if one_turn_result.current_node.source != 'NONE':
                ans_node = DialogueTurn.find_by_ae_id(one_turn_result.ans)

                if ans_node != "NONE":

                    # ノードがすでに追加されているか確認（ae_id で確認）
                    if not self.real_tree.has_node(ans_node.ae_id):
                        node_label = f"{ans_node.index + 1}: {self.wrap_and_truncate_text(ans_node.sentence)}"
                        self.real_tree.add_node(ans_node.ae_id, label=node_label, color=self.color_map[DialogueTurn.find_by_ae_id(ans_node.ae_id).speaker])

                    # エッジを追加（ae_id を用いる）
                    self.real_tree.add_edge(ans_node.ae_id, current_node.ae_id)

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
    def create_csv_header(out_put_dir_path):
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
            # 'tree_edit_distance',
            # 'path_based_similarity'
        ]

        #csvファイルを作成
        csv_file_path = os.path.join(out_put_dir_path, 'TreeResult.csv')
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
                # self.tree_edit_distance,
                # self.path_based_similarity
            ])

    @staticmethod
    def save_trees_from_result_class(result, tree_result_csv_path):
        tree_result = TreeResult(result)
        tree_result.save_one_tree(tree_result_csv_path)
        return tree_result



###################################################################################################
#以下は、グラフの描写に利用する処理

    # ノードのラベルを適切な長さで改行し、150文字を超える場合は省略する関数
    def wrap_and_truncate_text(self, text):
        width=40
        max_len=150
        if len(text) > max_len:
            text = text[:max_len] + '...'  # 150文字で切り捨てて省略記号を追加
        return '\n'.join(textwrap.wrap(text, width=width))

    # 話者ごとに異なる色を設定する関数
    def get_color_map(self, speakers):
        color_palette = ['lightblue', 'lightgreen', 'lightcoral', 'khaki', 'plum']
        color_map = {speaker: color_palette[i % len(color_palette)] for i, speaker in enumerate(speakers)}
        return color_map

    # 画像サイズをノード数に応じて計算する関数
    def calculate_figsize(self, num_nodes):
        if num_nodes <= 8:
            return (15, 12)  # デフォルトのサイズ
        else:
            # 8個以上のノードに対して、スケーリングするルール
            width = 15 + (num_nodes - 8) * 1.4  # 横幅を1.4ずつ増やす
            height = 12 + (num_nodes - 8) * 0.6  # 縦幅は0.6ずつ増やす
            return (width, height)

    # 凡例のスケーリング（フォントサイズとマーカーサイズを画像全体の大きさに基づいて設定）
    def calculate_legend_size(self, figsize):
        width, height = figsize
        # 画像全体の10%に相当するサイズを計算
        legend_font_size = max(30, 0.1 * width)
        legend_marker_size = max(300, 0.1 * width * 100)  # マーカーサイズはスケーリングを調整
        return legend_font_size, legend_marker_size

    def draw_tree(self, output_dir_path, type):


        if type == 'estimated':
            tree = self.estimated_tree
            output_dir = os.path.join(output_dir_path, 'estimated_tree_images')
        else:
            tree = self.real_tree
            output_dir = os.path.join(output_dir_path, 'real_tree_images')

        num_nodes = len(tree.nodes)
        figsize = self.calculate_figsize(num_nodes)  # 画像サイズを計算
        legend_font_size, legend_marker_size = self.calculate_legend_size(figsize)  # 凡例のサイズを計算

        # グラフを描画
        pos = graphviz_layout(tree, prog='dot')  # 'dot'レイアウトを使用
        plt.figure(figsize=figsize)  # 計算されたサイズを設定

        for node in tree.nodes:
            x, y = pos[node]
            if 'label' in tree.nodes[node]:  # labelが存在するか確認
                plt.text(x, y, tree.nodes[node]['label'], fontsize=10, ha='center', va='center',
                        bbox=dict(facecolor=tree.nodes[node]['color'], edgecolor='black', boxstyle='round,pad=0.3'))

        # エッジを描画
        nx.draw_networkx_edges(tree, pos, arrowstyle='-|>', arrowsize=15, edge_color='black')
        plt.axis('off')

        # 話者の凡例を描画（スケーリングされたフォントサイズとマーカーサイズを使用）
        for speaker, color in self.color_map.items():
            plt.scatter([], [], c=color, label=speaker, s=legend_marker_size)
        plt.legend(title='Speaker Colors', loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_font_size)

        # 画像をファイルに保存
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f'{self.file_name}.png')
        plt.savefig(output_path, format='png', bbox_inches='tight')
        plt.close()
