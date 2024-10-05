
import os
#/home/hasegawa_tomokazu/create_tree/CSV_topics配下のCSVファイルの数を取得する
def get_csv_files(csv_topics_path):
    csv_files = [f for f in os.listdir(csv_topics_path) if f.endswith('.csv')]
    return [os.path.join(csv_topics_path, csv_file) for csv_file in csv_files]

#CSV_topics配下のCSVファイルのパスを取得
csv_files = get_csv_files('/home/hasegawa_tomokazu/create_tree/CSV_topics')
print(len(csv_files))
