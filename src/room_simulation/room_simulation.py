# room_simulation.py
# Created by FreReRiku on 2024/12/29

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf
from scipy.io import wavfile

# music1かmusic2か選択する
music_type = 2

# スピーカーの数
num_spk = [0, 1]

# 各スピーカーに音源を割り当てる
channels = []
for spk in num_spk:
    fs, channel = wavfile.read(f'./../sound_data/original/music{music_type}_mono.wav')
    channels.append(channel)

# デバッグ用
print(f'サンプリング周波数：{fs}Hz')
print(f'チャンネル数：{len(channels)}ch')

# --部屋の設定----------
# 残響時間[s]
rt60 = 0.3
# 部屋の寸法[m]
room_dimensions = [3.52, 3.52, 2.4]

# Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)

# デバッグ用
print(f'壁のエネルギー吸収：{e_absorption}')
print(f'鏡像法での反射回数の上限：{max_order}回')

# 壁の材質を決める
m = pra.make_materials(
    ceiling =   "plasterboard",
    floor   =   "carpet_cotton",
    east    =   "plasterboard",
    south   =   "plasterboard",
    west    =   "plasterboard",
    north   =   "plasterboard",
)

# 部屋の材質
room = pra.ShoeBox(
    p           = room_dimensions,
    fs          = fs,
    materials   = m,
    max_order   = max_order
)

# マイク設置 [m]
mic_loc = [1.75, 1.75, 1.6]
room.add_microphone(mic_loc)

# スピーカーに座標情報を与える
room.add_source([3.4, 0.5, 0.5], signal=channels[0])
room.add_source([3.4, 2.3, 0.5], signal=channels[1])

# 部屋表示
fig, ax = room.plot()
ax.set_xlim([0, 3.6])
ax.set_ylim([0, 3.6])
ax.set_zlim([0, 2.5])

# シミュレーション & 保存

# --シミュレーション----------
# インパルス応答の計算
room.compute_rir()
# インパルス応答の保存
for i, ir_ in enumerate(room.rir):
    for j, ir in enumerate(ir_):
        ir_signal = ir
        ir_signal /= np.max(np.abs(ir_signal)) # 可視化のため正規化
        sf.write(f'../sound_data/room_simulation/impulse_signal_ch{j+1}_{fs}Hz.wav', ir_signal, fs)


# シミュレーション
separate_recordings = room.simulate(return_premix=True)

# 各音源のみを再生した音を保存
for i, sound in enumerate(separate_recordings):
    recorded        = sound[0, :]
    sf.write(f'../sound_data/room_simulation/music{music_type}_room_ch{num_spk[i]+1}_{fs}Hz.wav', recorded / np.max(recorded) * 0.95, fs)

# ミックスされた音源を保存
mixed_recorded  = np.sum(separate_recordings, axis=0)[0,:]
sf.write(f'../sound_data/room_simulation/music{music_type}_room_mix_{fs}Hz.wav', mixed_recorded / np.max(mixed_recorded) * 0.95, fs)

# 図示
plt.savefig('../figures/room_simulation/room.png')
