# make_data.py
# main.pyでの計算で得られたリストをCSV形式で書き出します.
# Created by FreReRiku on 2025/01/16

import numpy as np
import csv

def output_csv(input_array, item_name, output_file_name):

    """
    指定されたnumpy配列を読み込み, 新しいヘッダー行を追加してCSV形式で書き出します.

    Parameters
    ----------
    input_array : numpy.ndarray
        CSVファイルに書き込むNumPy配列. 配列の行がCSVの行になります.
    item_name : str
        出力ファイルに追加するヘッダー名.
    output_file_name : str
        データを保存する出力CSVファイルへのパス.
    
    Return
    ------
    None

    """

    # 入力配列が1次元の場合, 2次元配列に変換
    if input_array.ndim == 1:
        input_array = input_array.reshape(-1, 1)
    elif input_array.ndim != 2:
        raise ValueError("Input array must be 1-dimensional or 2-dimensional")

    # 出力ファイルにデータを書き込む
    with open(output_file_name, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([item_name])    # ヘッダー行
        writer.writerows(input_array.tolist())    # NumPy配列をリストに変換して書き込み.
    
    print(f"CSVファイルにデータが書き込まれました: {output_file_name}")
    return
