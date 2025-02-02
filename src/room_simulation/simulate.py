"""
simulate.py
----------

設定された音源や空間を用いてシミュレーションを行います.

Created by FreReRiku on 2025/01/17
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pyroomacoustics as pra
import soundfile as sf
from scipy.io import wavfile
import save
import convert
from pathlib import Path

def room(music_type):

    # ------------------------------
    # 音源・スピーカーの設定
    # ------------------------------
    directories = [
        Path("./../../sound/original"),
        Path("./../../sound/room_simulation"),
        Path("./../../figure/room_simulation"),
        Path("./../../data/room_simulation/room_info"),
        Path("./../../data/room_simulation/sr_and_spk")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 音源・スピーカーの設定
    # ------------------------------
    # 使用する音源の選択
    music_type = music_type
    # スピーカーの数
    num_spk = 2
    # 各スピーカーに音源を割り当てる
    channels = []
    for spk in range(num_spk):
        fs, channel = wavfile.read(f'./../../sound/original/music{music_type}_mono.wav')
        channels.append(channel)

    # サンプリング周波数とチャンネル数をCSV形式で書き出す
    save.sr_and_spk(fs, channels)

    # ------------------------------
    # 部屋の設定
    # ------------------------------
    # 残響時間[s]
    rt60 = 0.3
    # 部屋の寸法[m]
    room_dimensions = [3.52, 3.52, 2.4]
    # Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を求める (e_absorption, max_order)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)

    # テスト用: max_orderを強制的に5に設定
    max_order = 5

    # 部屋の情報をCSV形式で書き出す
    save.room_info(e_absorption, max_order)

    # # 壁の材質設定
    # m = pra.make_materials(
    #     ceiling =   "plasterboard",
    #     floor   =   "carpet_cotton",
    #     east    =   "plasterboard",
    #     south   =   "plasterboard",
    #     west    =   "plasterboard",
    #     north   =   "plasterboard",
    # )

    # 設定をroomに反映
    room = pra.ShoeBox(
        p           = room_dimensions,
        t0          = 0.0,
        fs          = fs,
        materials   = pra.Material(e_absorption),
        max_order   = max_order
    )

    # マイク設置 [m]
    mic_loc = [1.75, 1.75, 1.6]
    room.add_microphone(mic_loc)

    # スピーカーの座標情報
    room.add_source([3.4, 0.5, 0.5], signal=channels[0])
    room.add_source([3.4, 2.3, 0.5], signal=channels[1])

    # 部屋表示
    fig, ax = room.plot()
    ax.set_xlim([0, 3.6])
    ax.set_ylim([0, 3.6])
    ax.set_zlim([0, 2.5])

    # マイクとスピーカーを示す凡例用のハンドルを作成
    mic_handle = mlines.Line2D([], [], color='gray', marker='x', linestyle='None', markersize=5, label='Microphone')
    speaker1_handle = mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=5, label='Speaker1')
    speaker2_handle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=5, label='Speaker2')

    # axに凡例を追加
    ax.legend(handles=[mic_handle, speaker1_handle, speaker2_handle])

    # 画像の保存
    plt.savefig('./../../figure/room_simulation/room.png')
    plt.show()

    # 2D版の画像の保存
    # make_fig_2d(music_type)

    # ------------------------------
    # インパルス応答のシミュレーション
    # ------------------------------
    # 計算
    room.compute_rir()
    # 保存
    for i, ir_ in enumerate(room.rir):
        for j, ir in enumerate(ir_):
            ir_signal = ir
            ir_signal = ir_signal / np.max(np.abs(ir_signal))
            sf.write(f'./../../sound/room_simulation/impulse_signal_ch{j+1}_{fs}Hz.wav', ir_signal, fs)

    # ------------------------------
    # 音源を用いたシミュレーション
    # ------------------------------
    separate_recordings = room.simulate(return_premix=True)

    # --単体音声の保存----------
    for i, sound in enumerate(separate_recordings):
        recorded        = sound[0, :]
        sf.write(f'./../../sound/room_simulation/music{music_type}_room_ch{i+1}_{fs}Hz.wav', recorded / np.max(recorded) * 0.95, fs)

    # --混合音声の保存----------
    mixed_recorded  = np.sum(separate_recordings, axis=0)[0,:]
    sf.write(f'./../../sound/room_simulation/music{music_type}_room_mix_{fs}Hz.wav', mixed_recorded / np.max(mixed_recorded) * 0.95, fs)

    # --サンプリングレートの変更----------
    # 音源リスト
    sound_files = [
        './../../sound/room_simulation/impulse_signal_ch1',
        './../../sound/room_simulation/impulse_signal_ch2',
        f'./../../sound/room_simulation/music{music_type}_room_ch1',
        f'./../../sound/room_simulation/music{music_type}_room_ch2',
        f'./../../sound/room_simulation/music{music_type}_room_mix'
    ]

    # 変更したいサンプリングレート
    # target_sr = 16000
    # 
    # サンプリングレートの変更
    # for sound_file in sound_files:
    #     convert.sampling_rate(input_file=sound_file, orig_sr=fs, target_sr=target_sr)

    return

def make_fig_2d(music_type):

    # ------------------------------
    # 音源・スピーカーの設定
    # ------------------------------
    # 使用する音源の選択
    music_type = music_type
    # スピーカーの数
    num_spk = 2
    # 各スピーカーに音源を割り当てる
    channels = []
    for spk in range(num_spk):
        fs, channel = wavfile.read(f'./../../sound/original/music{music_type}_mono.wav')
        channels.append(channel)

    # ------------------------------
    # 部屋の設定
    # ------------------------------
    # 残響時間[s]
    rt60 = 0.3
    # 部屋の寸法[m]
    room_dimensions = [3.52, 3.52]
    # Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を求める (e_absorption, max_order)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)

    # テスト用: max_orderを強制的に5に設定
    max_order = 5

    # 設定をroomに反映
    room = pra.ShoeBox(
        p           = room_dimensions,
        t0          = 0.0,
        fs          = fs,
        materials   = pra.Material(e_absorption),
        max_order   = max_order
    )

    # マイク設置 [m]
    mic_loc = [1.75, 1.75]
    room.add_microphone(mic_loc)

    # スピーカーの座標情報
    room.add_source([3.4, 0.5], signal=channels[0])
    room.add_source([3.4, 2.3], signal=channels[1])

    # 部屋表示
    fig, ax = room.plot()
    ax.set_xlim([0, 3.6])
    ax.set_ylim([0, 3.6])

    # 画像の保存
    plt.savefig('./../../figure/room_simulation/room_2d.png')


    return
