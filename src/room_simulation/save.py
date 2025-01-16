"""
save.py
----------

値をCSV形式で出力するためのモジュールです.

Created by FreReRiku on 2024/12/29
"""

import csv
import os

file_path = './../../data/room_simulation'

def sr_and_spk(fs, channels):
    '''
    sr_and_spk
    ----------

    サンプリング周波数とチャンネル数をCSV形式で出力します。

    Parameter
    ---------
    fs: サンプリング周波数
    channels: チャンネル数

    Return
    ----------
    この関数は何も返しません。
    '''

    # 保存先ディレクトリを作成
    output_dir = os.path.join(file_path, 'sr_and_spk')
    os.makedirs(output_dir, exist_ok=True)

    # ファイルパスを指定
    output_file = os.path.join(output_dir, 'sr_and_spk.csv')

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['サンプリング周波数 [Hz]', 'チャンネル数 [ch]'])
        writer.writerow([fs, len(channels)])

    return

def room_info(e_absorption, max_order):
    '''
    room_info
    ----------

    壁のエネルギー吸収率と鏡像法での反射回数の上限をコンソールに出力します。

    Parameter
    ----------
    e_absorption: 壁面の平均吸音率
    max_order: 鏡像法での反射回数の上限

    Return
    ----------
    この関数は何も返しません。
    '''

    # 保存先ディレクトリを作成
    output_dir = os.path.join(file_path, 'room_info')
    os.makedirs(output_dir, exist_ok=True)

    # ファイルパスを指定
    output_file = os.path.join(output_dir, 'room_info.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['壁のエネルギー吸収率', '鏡像法での反射回数の上限 [回]'])
        writer.writerow([e_absorption, max_order])
    
    return
