# room_simulation_no_embedded.py

# import (settingsはsrcディレクトリ内のsettings.pyを使用)
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf
from scipy.io import wavfile
from settings import *

# wavファイル読み込み
# seedsの数=スピーカーの数なので、この場合は2個のスピーカーから音源を流すよう指定している。

# music1かmusic2か選択する
# 以下の値に1か2を入力してください

music_type = 2

print('\nroom_simulation.pyを実行します。')
print('以下に結果を表示します（デバッグ用）')

channels = []
for seed in seeds:
    fs, channel = wavfile.read(f'./../sound_data/original_sound_source/music{music_type}_mono.wav')
    channels.append(channel)

print(f'サンプリング周波数：{fs}Hz')
print(f'チャンネル数：{len(channels)}ch')

# 部屋の設定
# 残響時間と部屋の寸法
rt60 = 0.3  # 残響時間[s]
room_dimensions = [3.52, 3.52, 2.4]  # 部屋の寸法[m] ここを二次元にすると二次平面の部屋になります

# Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
e_absorption, max_order = pra.inverse_sabine(rt60, room_dimensions)

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

# 部屋をつくります
# fsは生成されるインパルス応答のサンプリング周波数です。入力する音源があるならそれに合わせる。
room = pra.ShoeBox(
    room_dimensions,
    fs          =   fs,
    materials   =   m, # 変更前: pra.Material(e_absorption)
    max_order   =   max_order # 変更前: 5
)

# マイク設置
mic_loc = [1.75, 1.75, 1.6]
room.add_microphone(mic_loc)

# 音源ごとに座標情報を与え、`room`に追加していきます。
# オプションで delay を追加することもできます。
room.add_source([3.4, 0.5, 0.5], signal=channels[0])
room.add_source([3.4, 2.3, 0.5], signal=channels[1])

# 部屋表示
fig, ax = room.plot()
ax.set_xlim([0, 3.6])
ax.set_ylim([0, 3.6])
ax.set_zlim([0, 2.5])

# シミュレーション & 保存

# インパルス応答を計算
room.compute_rir()
# インパルス応答を保存

for i, ir_ in enumerate(room.rir):
    for j, ir in enumerate(ir_):
        ir_signal = ir
        ir_signal /= np.max(np.abs(ir_signal)) # 可視化のため正規化
        sf.write(f'../sound_data/room_simulation/impulse_mic{i+1}_ch{j+1}.wav', ir_signal, fs)


# シミュレーション
separate_recordings = room.simulate(return_premix=True)
print(type(separate_recordings))
print(separate_recordings)

# 各音源のみを再生した音を保存
for i, sound in enumerate(separate_recordings):
    print(list(enumerate(separate_recordings)))
    print(i)
    recorded        = sound[0, :]
    sf.write(f'../sound_data/room_simulation/music{music_type}_room_seed{seeds[i]}.wav',
             recorded / np.max(recorded) * 0.95,
             fs)

# ミックスされた音源を保存
mixed_recorded  = np.sum(separate_recordings, axis=0)[0,:]
sf.write(f'../sound_data/room_simulation/music{music_type}_room_mix.wav', mixed_recorded / np.max(mixed_recorded) * 0.95, fs)

# 図示
plt.show()

