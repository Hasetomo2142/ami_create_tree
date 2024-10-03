import csv
import json
import os
import re

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
csv_topics_path = os.path.join(dir_path, 'CSV_topics')

class DialogueTurn:
    def __init__(self, index, ae_id, speaker, start_time, end_time, sentence, source, targets):
        self.index = index
        self.ae_id = ae_id
        self.speaker = speaker
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.sentence = sentence
        self.source = source
        self.targets = self.parse_targets(targets)

    # CSVから情報を取得し、DialogueTurnのリストを返す。
    @staticmethod
    def from_csv(csv_path):
        dialogue_turns = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # ヘッダー行をスキップ
            for index, row in enumerate(reader):
                if len(row) >= 7:
                    # コンストラクタの順番で変数を格納
                    ae_id = row[0]
                    speaker = row[1]
                    start_time = row[2]
                    end_time = row[3]
                    sentence = row[4]
                    source = row[5]
                    targets = row[6]
                    dialogue_turns.append(DialogueTurn(index, ae_id, speaker, start_time, end_time, sentence, source, targets))
                else:
                    print(f"行のフォーマットが正しくありません: {row}")
        return dialogue_turns

    def parse_targets(self, targets_str):
        if targets_str == ['NONE'] or not targets_str:
            return []
        else:
            return [t.strip() for t in targets_str.strip('{}').split(',')]

    # 与えられたsourceのae_idとtargetのae_idが関連しているかどうかを確認するメソッド
    @staticmethod
    def relationship_exists(dialogue_turns, source_ae_id, target_ae_id):
        source_ae_id.strip()
        target_ae_id.strip()
        for turn in dialogue_turns:
            if turn.ae_id == source_ae_id and target_ae_id == turn.source:
                return True
        return False

    # sourceとtargetsがNoneまたは空であるノードを削除するメソッド
    @staticmethod
    def remove_none_relationships(dialogue_turns):
        filtered_turns = []
        removed_turns = []
        for turn in dialogue_turns:
            if turn.source == 'NONE' and turn.targets == ['NONE']:
                removed_turns.append(turn)
            else:
                filtered_turns.append(turn)
        return filtered_turns, removed_turns


    # ae_idからファイルを検索し、ファイルパスのリストを返すメソッド
    @staticmethod
    def find_by_ae_id(ae_id):
        # 正規表現パターンを定義
        pattern = r"^[^.]+"  # ドットまでの部分を抽出
        # ae_idからファイルIDを抽出
        match = re.match(pattern, ae_id)
        file_id = match.group()

        # マッチするファイルパスを保存するリスト
        matching_files = []

        # dir_pathのなかで、file_idから始まるファイルのリストを検索
        for file_name in os.listdir(csv_topics_path):
            if file_name.startswith(file_id):
                file_path = os.path.join(csv_topics_path, file_name)
                matching_files.append(file_path)  # 見つかったファイルをリストに追加

        for file_path in matching_files:
            dialogue_turns = DialogueTurn.from_csv(file_path)
            for turn in dialogue_turns:
                if turn.ae_id == ae_id:
                    return turn
        return 'NONE'

    # 自身の辞書型を返すメソッド
    def to_dict(self):
        return {
            "index": self.index,
            "ae_id": self.ae_id,
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sentence": self.sentence,
            "source": self.source,
            "targets": self.targets
        }


    @classmethod
    def from_dict(cls, data):
        return cls(
            index=data["index"],
            ae_id=data["ae_id"],
            speaker=data["speaker"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            sentence=data["sentence"],
            source=data["source"],
            targets=data["targets"]
        )

# dir_path = "/home/hasegawa_tomokazu/create_tree/CSV_topics"  # 検索するディレクトリのパス
# ae_id = "ES2002a.D.argumentstructs.Erik.1"
# node = DialogueTurn.find_by_ae_id(dir_path, ae_id)
# print(node.to_dict())
