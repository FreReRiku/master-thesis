'''----------
display.py
Created by FreReRiku on 2024/12/29
----------'''

def sr_and_spk(fs, channels):
    '''
    sr_and_spk
    ----------

    サンプリング周波数とチャンネル数をコンソールに出力します。

    Parameter
    ---------
    fs: サンプリング周波数
    channels: チャンネル数

    Return
    ----------
    この関数は何も返しません。
    '''

    print(f'サンプリング周波数：{fs}Hz')
    print(f'チャンネル数：{len(channels)}ch')
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

    print(f'壁のエネルギー吸収：{e_absorption}')
    print(f'鏡像法での反射回数の上限：{max_order}回')
    return
