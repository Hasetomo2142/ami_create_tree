import json
import os

# 必要なクラスをインポートまたは定義済みと仮定
from classes.dialogue_turn import DialogueTurn
from classes.result_json import OneTurnResult, Result

# ファイルのパス
dir_path = '/home/hasegawa_tomokazu/create_tree/'
result_json_path = os.path.join(dir_path, 'result_json')


# 使用例
if __name__ == "__main__":
    # 読み込むJSONファイルのパスを指定
    json_file_path = '/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.0.txt/ES2002a-ES2002a - Regions.json'

    # JSONファイルからResultオブジェクトを復元
    result = Result.load_result_from_json(json_file_path)

    print(result.to_json())  # JSON形式で出力
