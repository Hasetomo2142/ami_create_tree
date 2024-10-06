import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# CSVファイルを1行ずつ読み込む関数（再利用）
def read_csv_file(file_path):
    """指定されたCSVファイルを1行ずつ読み込む。1行目はヘッダーとして読み飛ばす"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーを読み飛ばす
        for row in reader:
            yield row

# ％表示された文字列をfloat型の数値に変換する
def percentage_to_float(percentage_str, index):
    """％表示の文字列をfloatに変換し、100%を超える場合は例外を発生させる"""
    try:
        print(percentage_str)
        percentage = float(percentage_str.strip('%')) / 100.0
        print(percentage)
        print('-----------------')
        if percentage > 1.0:
            raise ValueError(f"Invalid percentage value: {percentage_str} at index {index}. Accuracy cannot exceed 100%.")
        return percentage
    except ValueError:
        raise ValueError(f"Could not convert percentage value: {percentage_str} at index {index}")

# 正解率と木の深さを抽出して返す関数
def extract_accuracy_and_depth(file_path):
    """正解率と木の深さを抽出し、それぞれのリストとして返す"""
    accuracies = []
    depths = []
    error_indices = []

    for i, row in enumerate(read_csv_file(file_path)):
        try:
            accuracy = percentage_to_float(row[1], i)  # index=2の正解率を変換
            depth = int(row[5])  # index=6の木の深さをintに変換
            accuracies.append(accuracy)
            depths.append(depth)
        except ValueError as e:
            print(e)
            error_indices.append(i)

    return accuracies, depths, error_indices

# 正解率と木の深さの相関関係をプロットする関数
def plot_accuracy_vs_depth(accuracies, depths, output_image_path):
    """正解率と木の深さの相関関係をプロットし、相関係数を表示して画像として保存する"""

    # 相関係数を計算（Pearsonの相関係数）
    if len(accuracies) > 1 and len(depths) > 1:
        correlation_coefficient, _ = pearsonr(accuracies, depths)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=accuracies, y=depths, s=100)  # 点の大きさを大きくするためにs=100を追加
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Tree Depth")
        plt.title(f"Correlation between Accuracy and Tree Depth\nCorrelation Coefficient: {correlation_coefficient:.2f}")

        # プロットを画像として保存
        plt.savefig(output_image_path)
        plt.close()

        # 相関係数を返す
        return correlation_coefficient
    else:
        print("Not enough valid data points to calculate correlation.")
        return None

# メイン処理
if __name__ == '__main__':
    csv_file_path = '/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.1-4o-mini.txt/TreeResult.csv'
    output_image_path = '/home/hasegawa_tomokazu/create_tree/accuracy_vs_depth.png'

    # 正解率と木の深さを抽出
    accuracies, depths, error_indices = extract_accuracy_and_depth(csv_file_path)

    # エラーを引き起こしたデータのインデックスを表示
    if error_indices:
        print(f"Data errors occurred at indices: {error_indices}")

    # 相関関係をプロットし、相関係数を表示
    correlation_coefficient = plot_accuracy_vs_depth(accuracies, depths, output_image_path)

    if correlation_coefficient is not None:
        # 相関係数を標準出力
        print(f"Correlation Coefficient: {correlation_coefficient:.2f}")
    else:
        print("No valid data to plot.")
