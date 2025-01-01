'''----------
output.py
結果をファイルに出力します.
Created by FreReRiku on 2024/12/31
----------'''
import csv

# --CSV出力用関数--------------------
def output_csv(file_name, data, mode):
    '''
    output_csv
    データをCSV形式で書き込みます.

    Parameter
    ----------
    file_name: ファイル名を入力してください(任意の名前)
    data: 書き込むファイルを選択してください
    mode: ファイルに書き出すときのモード設定

    Return
    ----------
    この関数には戻り値がありません.
    '''

    with open(f'./../../data/distance_estimation/{file_name}', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # データを書き込む
        writer.writerows(data)
    
    return
