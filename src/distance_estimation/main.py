# distance_estimation.py
# 埋込強度の変化に伴う推定誤差結果の変化をプロット
# Created by FreReRiku on 2024/12/30

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import csv
from pesq import pesq
from librosa import stft, istft, resample
from scipy.signal import find_peaks
from scipy.fft import irfft

# ------------------------------
# パラメータ
# ------------------------------

# 音源の選択 (1 or 2)
music_type      = 1
# サンプリング周波数 [Hz]
sampling_rate   = 44100
# 音速 [m/s]
speed_of_sound  = 340.29
# サンプル長 [sample]
signal_length_samples   = sampling_rate * 10
# フレーム長 [sample]
frame_length    = 1024
# ホップ長 [sample]
hop_length      = 512

# スタートポイント [sample]
start_sample    = 1000
# エンドポイント   [sample]
end_sample      = start_sample + signal_length_samples

# トライアル回数 [times]
num_trials      = 100

# 連続して埋め込むフレーム数 [sample]
num_embedding_frames        = 40
# 埋め込み周波数のビン数 [sample]
embedding_frequency_bins    = np.floor(frame_length*0.1).astype(int)
# スタートのフレーム位置(ここからKフレーム用いる)
frame_count = round(sampling_rate*3/16000)     # フレームカウント
pos_st_frame = np.arange(0, num_trials*frame_count, frame_count)
# CSPの最大値に対するノイズと判定する振幅比のしきい値(const)
threshold_ratio  = 0.2

# -埋め込む振幅の設定-----
# ループ回数 [times]
loop_times = 25
# 埋め込む振幅
embedding_amplitudes    = np.linspace(0, 1, loop_times)
# -埋め込む位相の設定-----
embedding_phase  = 0

# ゼロ除算回避定数 (const)
epsilon = 1e-20

# -データ格納用のリストの初期化-----
# 遅延推定誤差記録用
delay_time_errors = []
# 音質評価記録用
pesq_scores = []
# スピーカー1とマイク間の距離・到来時間の記録用
distance_speaker1 = []
# スピーカー2とマイク間の距離・到来時間の記録用
distance_speaker2 = []

# ------------------------------
# ファイル出力
# ------------------------------

# 初期条件の出力
with open(f'./../../data/distance_estimation/music{music_type}_mono/init_information.csv', mode='w', newline='', encoding='utf-8') as file_init_information:
    writer = csv.writer(file_init_information)
    writer.writerows( [
    [f'{frame_length+1}binのうちゼロを埋め込む周波数ビンの数[bin]','1回の検知で埋め込むフレーム数[フレーム]','試行回数[回]'],
    [f'{embedding_frequency_bins}',f'{num_embedding_frames}',f'{len(pos_st_frame)}']
])

# スピーカー1におけるマイク・スピーカー間距離、到来時間の出力
with open(f'./../../data/distance_estimation/music{music_type}_mono/distance_and_arrival_spk1.csv', mode='w', newline='', encoding='utf-8') as file_distance_and_arrival_spk1:
    writer = csv.writer(file_distance_and_arrival_spk1)
    writer.writerow(['マイク・スピーカ間距離[m]','到来時間[ms]'])

# スピーカー2におけるマイク・スピーカー間距離、到来時間の出力
with open(f'./../../data/distance_estimation/music{music_type}_mono/distance_and_arrival_spk2.csv', mode='w', newline='', encoding='utf-8') as file_distance_and_arrival_spk2:
    writer = csv.writer(file_distance_and_arrival_spk2)
    writer.writerow(['マイク・スピーカ間距離[m]','到来時間[ms]'])

# ピークレシオ・検知確率の出力
with open(f'./../../data/distance_estimation/music{music_type}_mono/peak_ratio.csv', mode='w', newline='', encoding='utf-8') as file_peak_ratio:
    writer = csv.writer(file_peak_ratio)
    writer.writerow(['平均Peak Ratio','最小Peak Ratio','正しく検知できる確率[%]'])

# PESQ、SN比の出力
with open(f'./../../data/distance_estimation/music{music_type}_mono/pesq.csv', mode='w', newline='', encoding='utf-8') as file_pesq:
    writer = csv.writer(file_pesq)
    writer.writerow(['PESQ','SNR[dB]'])


for num, amp in enumerate(embedding_amplitudes):
    # ------------------------------
    # オーディオファイルの読み込み
    # ------------------------------

    # ファイルパスの指定
    file_name_impulse1  = f'./../../sound/room_simulation/impulse_signal_ch1_{sampling_rate}Hz.wav'
    file_name_impulse2  = f'./../../sound/room_simulation/impulse_signal_ch2_{sampling_rate}Hz.wav'
    file_name_origin    = f'./../../sound/original/music{music_type}_mono.wav'
    file_name_origin_long    = f'./../../sound/original/long_music{music_type}_mono.wav'
    file_name_spk1 = f'./../../sound/room_simulation/music{music_type}_room_ch1_{sampling_rate}Hz.wav'
    file_name_spk2 = f'./../../sound/room_simulation/music{music_type}_room_ch2_{sampling_rate}Hz.wav'
    file_name_spk_long = f'./../../sound/room_simulation/long_music{music_type}_room_ch1_{sampling_rate}Hz.wav'
    # 読み込み
    impulse1, _ = sf.read(file_name_impulse1)
    impulse2, _ = sf.read(file_name_impulse2)
    x, _        = sf.read(file_name_origin)
    xlong, _    = sf.read(file_name_origin_long)
    y1, _       = sf.read(file_name_spk1)
    y2, _       = sf.read(file_name_spk2)
    ylong, _    = sf.read(file_name_spk_long)

    # スペクトログラム
    Xspec   = stft(x, n_fft=2*frame_length, hop_length=hop_length, win_length=frame_length, center=False)
    Y1spec  = stft(y1, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)
    Y2spec  = stft(y2, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)
    Y1zero  = stft(ylong, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)

    # 保存用の配列
    csp0_values, csp1_values, csp2_values, embedded_freq_csp1_values, embedded_freq_csp2_values,  csp_difference_values, weighted_csp_values, CSP_emb_sub_data, CSP_emb_wtd_data = [], [], [], [], [], [], [], [], []


    # ------------------------------
    # 1st: CSP0, 及び 遅延距離d_0 の推定
    # ------------------------------
    for frame_start_index in pos_st_frame:

        # マイク入力音声のスペクトログラム(スペクトログラムの合成)
        Yspec = Y1spec + Y2spec

        # --CSP0を求める(GCC-PHAT法)----------
        # 相互相関(周波数領域)
        XY0 = Yspec[:, frame_start_index:frame_start_index + num_embedding_frames] * np.conj(Xspec[:, frame_start_index:frame_start_index + num_embedding_frames])
        # 相互相関の絶対値(周波数領域)
        XY0abs = np.abs(XY0)
        # 分母がほぼ0になるのを防止
        XY0abs[XY0abs < epsilon] = epsilon
        # 白色化相互相関(周波数領域)
        CSP0_sp = XY0 / XY0abs
        # 時間方向で平均
        CSP0 = np.mean(CSP0_sp, axis=1)
        # 逆STFT
        CSP0_ave = irfft(CSP0, axis=0)

        # -CSP0の整形-----
        # いらない後半部を除去
        CSP0_ave = CSP0_ave[:frame_length]
        # 最大で割り算
        CSP0_ave = CSP0_ave / np.max(CSP0_ave)

        # -CSP0の保存-----
        csp0_values.append(CSP0_ave)

        # -dを推定-----
        estimated_delay = (np.argmax(csp0_values)-67) # この67という値を変えるとpeak_ratioの結果に影響が出る. 最適な値を動的に導出できるようにしたい

    # --インパルスのピーク位置の推定----------
    # インパルス応答ピークの記録用
    impulse_peak_positions = []
    # 遅延時間を除いたインパルス応答ピークの記録用
    adjusted_impulse_peaks = []

    # find_peaks関数を用いて、ピークを推定・記録
    for impulse_ in [impulse1, impulse2]:
        # scipy.signal.find_peaks関数を使って, impulse_からピークを検出
        pos_peaks, _ = find_peaks(impulse_, height=0.6)     # height: ピーク位置の閾値
        impulse_peak_positions.append(pos_peaks[0])
        adjusted_impulse_peaks.append(pos_peaks[0]-estimated_delay)

    # numpyに変換
    impulse_peak_positions = np.array(impulse_peak_positions)
    adjusted_impulse_peaks = np.array(adjusted_impulse_peaks)

    # print(impulse_peak_positions)
    # print(adjusted_impulse_peaks)

    # 遅延時間を考慮した音声のトリミング
    adjusted_x       = xlong[start_sample-estimated_delay:end_sample-estimated_delay]

    # Xspecを更新し、遅延時間を考慮したスペクトログラムを生成
    Xspec_adjusted   = stft(adjusted_x, n_fft=2*frame_length, hop_length=hop_length, win_length=frame_length, center=False)

    for frame_start_index in pos_st_frame:

        # 埋め込み用の配列
        Y1emb   = np.copy(Y1spec)

        # マイク入力音声のスペクトログラム
        Yspec       = Y1spec + Y2spec

        # ------------------------------
        # 2nd: CSP1を求める
        # ------------------------------
        # 相互相関(周波数領域)
        XY1          = Yspec[:, frame_start_index:frame_start_index+num_embedding_frames] * np.conj(Xspec_adjusted[:, frame_start_index:frame_start_index+num_embedding_frames])
        # 相互相関の絶対値(周波数領域)
        XY1abs       = np.abs(XY1)
        # 分母がほぼ0になるのを防止
        XY1abs[XY1abs < epsilon] = epsilon
        # 白色化相互相関(周波数領域)
        CSP1_sp      = XY1/XY1abs
        # 時間方向で平均
        CSP1         = np.mean(CSP1_sp, axis=1)
        # 逆STFT
        CSP1_ave     = irfft(CSP1, axis=0)

        # -CSP0の整形-----
        # いらない後半部を除去
        CSP1_ave     = CSP1_ave[:frame_length]
        # 最大で割り算
        CSP1_ave     = CSP1_ave/np.max(CSP1_ave)

        # --ゼロ埋め込み周波数の決定----------
        # 振幅(周波数?)の大きい順にインデックスを取得
        pos         = np.argsort(-np.abs(CSP1))
        # CSPの最大embedding_frequency_bins個の周波数
        embedded_freq   = pos[:embedding_frequency_bins]

        # --CSP1(埋込周波数のみ)を求める. CSP1の特定の周波数成分だけを抽出し, 時間領域に変換----------
        # CSP1と同じ形状の配列を作成し, すべての値を0に初期化
        CSP1_emb = np.zeros_like(CSP1)
        # embedded_freqに指定された周波数成分だけをCSP1_embにコピーし, それ以外を0にする
        CSP1_emb[embedded_freq]     = CSP1[embedded_freq]
        # 特定の周波数成分だけを含む信号を時間領域に変換する
        CSP1_emb_ave     = irfft(CSP1_emb, axis=0)

        # -CSP1の整形-----
        # いらない後半部を除去
        CSP1_emb_ave     = CSP1_emb_ave[:frame_length]
        # 最大で割り算
        CSP1_emb_ave     = CSP1_emb_ave/np.max(CSP1_emb_ave)

        # ------------------------------
        # 3rd: 振幅変調と位相変調
        # ------------------------------
        # Y1 に対して振幅変調
        Y1emb[embedded_freq, :] = amp * Y1emb[embedded_freq, :]        # embedded_freqの周波数ビンにamp倍
        # Y1 に対して位相変調
        theta = embedding_phase/180 * np.pi
        Y1emb[embedded_freq, :] = Y1emb[embedded_freq, :] * np.exp(1j*theta)
        # print(f'Y1emb shape: {Y1emb.shape}')

        # -音質検査用-----
        Y1zero[embedded_freq, frame_start_index:frame_start_index+3] = amp * Y1zero[embedded_freq, frame_start_index:frame_start_index+3]

        # ------------------------------
        # 4th: CSP2を求める
        # ------------------------------
        # 埋め込み信号を利用している(Y1emb)
        Yspec   = Y1emb + Y2spec
        # 相互相関(周波数領域)
        XY2         = Yspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames] * np.conj(Xspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames])
        # 相互相関の絶対値(周波数領域)
        XY2abs       = np.abs(XY2)
        # 分母がほぼ0になるのを防止
        XY2abs[XY2abs < epsilon] = epsilon
        # 白色化相互相関(周波数領域)
        CSP2_sp      = XY2 / XY2abs
        # 時間方向で平均
        CSP2         = np.mean(CSP2_sp, axis=1)
        # 逆STFT
        CSP2_ave     = irfft(CSP2, axis=0)

        # -CSP2の整形-----
        # いらない後半部を除去
        CSP2_ave     = CSP2_ave[:frame_length]
        # 最大で割り算
        CSP2_ave     = CSP2_ave/np.max(CSP2_ave)

        # --CSP2(埋込周波数のみ)を求める. CSP2の特定の周波数成分だけを抽出し, 時間領域に変換----------
        # CSP2と同じ形状の配列を作成し, すべての値を0に初期化
        CSP2_emb = np.zeros_like(CSP2)
        # embedded_freqに指定された周波数成分だけをCSP2_embにコピーし, それ以外を0にする
        CSP2_emb[embedded_freq]     = CSP2[embedded_freq]
        # 特定の周波数成分だけを含む信号を時間領域に変換する
        CSP2_emb_ave = irfft(CSP2_emb, axis=0)

        # -CSP2(埋込周波数のみ)の整形-----
        # いらない後半部を除去
        CSP2_emb_ave = CSP2_emb_ave[:frame_length]
        # 最大で割り算
        CSP2_emb_ave = CSP2_emb_ave / np.max(CSP2_emb_ave)

        # ------------------------------
        # 5th: 重み付き差分CSPを求める
        # ------------------------------
        # -重みを計算する-----
        # CSPのピーク位置を計算
        pk_csp, _  = find_peaks(CSP1_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        index      = np.argsort(-CSP1_ave[pk_csp])
        # CSPの大きい順にembedding_frequency_bins位のピーク位置をピークの大きい順に取得
        pk_csp     = pk_csp[index[:embedding_frequency_bins]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1     = pk_csp[0]

        # 重み
        weight      = np.copy(CSP1_ave)
        # 推定した第１スピーカのピークを除去
        weight[delay1-3:delay1+3] = 0
        # 閾値以下の値を0にする
        weight[weight < threshold_ratio]   = 0
        # 正規化
        weight      = weight/np.max(np.abs(weight))


        # ------------------------------
        # 6th: 重み付け差分CSPによる遅延推定
        # ------------------------------
        # CSPの差分
        CSP_sub     = CSP1_ave - CSP2_ave
        # 正規化
        CSP_sub     = CSP_sub / np.max(CSP_sub)

        # 重み付け埋込差分CSP
        CSP_wt_sub  = weight*CSP_sub

        # ------------------------------
        # 7th: 重み付け差分CSP(埋込周波数のみ)用の重み計算
        # ------------------------------
        # CSPのピーク位置を計算
        pk_csp, _  = find_peaks(CSP1_emb_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        index      = np.argsort(-CSP1_emb_ave[pk_csp])
        # CSPの大きい順にembedding_frequency_bins位のピーク位置をピークの大きい順に取得
        pk_csp     = pk_csp[index[:embedding_frequency_bins]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1     = pk_csp[0]

        # 重み
        weight = np.copy(CSP1_emb_ave)
        # 推定した第１スピーカのピークを除去
        weight[delay1 - 3:delay1 + 3] = 0
        # 閾値以下の値を除去
        weight[weight < threshold_ratio] = 0
        # 正規化
        weight = weight / np.max(np.abs(weight))

        # ------------------------------
        # 8th: 重み付け差分CSP(埋込周波数のみ)による遅延推定
        # ------------------------------
        # CSPの差分
        CSP_emb_sub     = CSP1_emb_ave - CSP2_emb_ave
        # 正規化
        CSP_emb_sub     = CSP_emb_sub / np.max(CSP_emb_sub)

        # 重み付け埋込差分CSP
        CSP_emb_wt      = weight*CSP_emb_sub

        # ------------------------------
        # 9th: 計算結果を保存する
        # ------------------------------
        csp1_values.append(CSP1_ave)
        csp2_values.append(CSP2_ave)
        embedded_freq_csp1_values.append(CSP1_emb_ave)
        embedded_freq_csp2_values.append(CSP2_emb_ave)
        csp_difference_values.append(CSP_sub)
        weighted_csp_values.append(CSP_wt_sub)
        CSP_emb_sub_data.append(CSP_emb_sub)
        CSP_emb_wtd_data.append(CSP_emb_wt)

    # numpyに変更
    csp1_values                 = np.array(csp1_values)                 # CSP1
    csp2_values                 = np.array(csp2_values)                 # CSP2
    embedded_freq_csp1_values   = np.array(embedded_freq_csp1_values)   # CSP1(埋込周波数のみ)
    embedded_freq_csp2_values   = np.array(embedded_freq_csp2_values)   # CSP2(埋込周波数のみ)
    csp_difference_values       = np.array(csp_difference_values)       # 差分CSP
    weighted_csp_values         = np.array(weighted_csp_values)         # 重み付き差分CSP
    CSP_emb_sub_data            = np.array(CSP_emb_sub_data)            # 差分CSP
    CSP_emb_wtd_data            = np.array(CSP_emb_wtd_data)            # 重み付き差分CSP

    # 推定誤差を算出
    distance_speaker1 = [f'{impulse_peak_positions[0]/sampling_rate*speed_of_sound:.2f},{1000*impulse_peak_positions[0]/sampling_rate:.2f}']
    distance_speaker2 = [f'{impulse_peak_positions[1]/sampling_rate*speed_of_sound:.2f},{1000*impulse_peak_positions[1]/sampling_rate:.2f}']
    with open(f'./../../data/distance_estimation/music{music_type}_mono/distance_and_arrival_spk1.csv', mode='a', newline='', encoding='utf-8') as file_distance_and_arrival_spk1:
        writer = csv.writer(file_distance_and_arrival_spk1)

        for entry in distance_speaker1:
            dist_m, dist_mm = entry.split(',')
            writer.writerow([dist_m, dist_mm])

    with open(f'./../../data/distance_estimation/music{music_type}_mono/distance_and_arrival_spk2.csv', mode='a', newline='', encoding='utf-8') as file_distance_and_arrival_spk2:
        writer = csv.writer(file_distance_and_arrival_spk2)

        for entry in distance_speaker2:
            dist_m, dist_mm = entry.split(',')
            writer.writerow([dist_m, dist_mm])

    # ------------------------------
    # 10th: 遅延量推定精度を求める
    # ------------------------------

    # csp1_valuesとCSP_emb_wtd_dataに基づいて, 遅延量(delay)を推定し,
    # その結果をリストDelayに格納する.
    Delay = []
    for csp1, csp2 in zip(csp1_values, CSP_emb_wtd_data):
        # 遅延量の推定
        csp1_imp = []
        csp1_peaks, _ = find_peaks(csp1, height=0.5)
        # 最初に検出されたピークの位置(csp1_peaks[0])をcsp1_impに追加
        csp1_imp.append(csp1_peaks[0])
        # csp1における最初のピーク位置(csp1_imp[0])をdelay1として保存する.
        delay1 = csp1_imp[0]
        # csp2の最大値を持つインデックスを取得し, それをdelay2として保存する.
        delay2 = np.argmax(csp2)
        # delay1(csp1のピーク位置)とdelay2(csp2の最大値位置)を配列として格納する.
        delay = np.array([delay1, delay2])
        Delay.append(delay)

    Delay = np.array(Delay)

    # 遅延推定誤差 (平均絶対誤差)
    error = []
    for delay in Delay:
        # delay(推定遅延量)とadjusted_impulse_peaks(基準値)の差の絶対値を計算し, それらを足し合わせる.
        tmp1 = np.sum(np.abs(delay - adjusted_impulse_peaks))
        # 遅延ペア([delay1, delay2])の順序が逆である場合の誤差を計算.
        # 遅延ペアの順序が異なっている場合の比較を考慮している.
        tmp2 = np.sum(np.abs(np.flip(delay) - adjusted_impulse_peaks))
        # tmp1, tmp2のうち, 小さい方の値(最小誤差)を選択して, リストerrorに追加.
        error.append(np.min([tmp1, tmp2]))
    # リストerrorに格納されたすべての遅延誤差の平均を計算し,
    # 全体としての平均的な遅延誤差を取得.
    error = np.mean(np.array(error))
    # サンプル数単位から時間単位へ変換
    mean_delay_error_ms = 1000 * (error / sampling_rate)
    delay_time_errors.append(mean_delay_error_ms)

    PR_data = []
    for csp2, delay in zip(CSP_emb_wtd_data, Delay):
        # まずcsp1が第１スピーカと第２スピーカどちらの遅延を検知したか判定
        # 結果をpos_truthに保存.
        if np.abs(delay[0] - adjusted_impulse_peaks[0]) < np.abs(delay[0] - adjusted_impulse_peaks[1]):
            pos_truth = adjusted_impulse_peaks[1]  # csp2はpos_imp[1]を推定したと判定
        else:
            pos_truth = adjusted_impulse_peaks[0]  # csp2はpos_imp[0]を推定したと判定

        # 真の遅延 pos_truth におけるピーク振幅を取得.
        csp2_peak = csp2[pos_truth]

        # それ以外での最大ピーク
        # csp2のコピーを作成する
        tmp = np.copy(csp2)
        # 真の遅延位置(pos_truth)を0に設定
        tmp[pos_truth] = 0
        # 他のピークの中で最大値を探す(np.max(tmp)).
        peak_2nd = np.max(tmp)

        # 真のピークが他のピークよりどれだけ際立っているかを数値化.
        PR_data.append(csp2_peak / (np.max([peak_2nd, 10 ** (-8)])))

    # numpy配列に変換
    PR_data = np.array(PR_data)

    Peak_Ratio = [f'{np.mean(PR_data):.2f},{np.min(PR_data):.2f},{(PR_data[PR_data >= 1].size / PR_data.size)*100:2.0f}']
    with open(f'./../../data/distance_estimation/music{music_type}_mono/peak_ratio.csv', mode='a', newline='', encoding='utf-8') as file_peak_ratio:
        writer = csv.writer(file_peak_ratio)
        for entry in Peak_Ratio:
            dist_m, dist_mm, dist_mmm = entry.split(',')
            writer.writerow([dist_m, dist_mm, dist_mmm])

    # ------------------------------
    # 11th: 音質評価
    # ------------------------------
    # 時間波形
    frames  = min([Y1spec.shape[1], Y1zero.shape[1]])
    y1_orig = istft(Y1spec[:,:frames], hop_length=hop_length)
    y1_emb  = istft(Y1zero[:,:frames], hop_length=hop_length)

    # PESQ
    y1_orig_ds = resample(y1_orig[:sampling_rate*5], orig_sr=sampling_rate, target_sr=sampling_rate)
    y1_emb_ds  = resample(y1_emb[:sampling_rate*5] , orig_sr=sampling_rate, target_sr=sampling_rate)
    score = pesq(16000, y1_orig_ds, y1_emb_ds)
    # SNR
    snr = 20 * np.log10(sum(y1_orig ** 2) / sum((y1_orig - y1_emb) ** 2))

    pesq_and_snr = [f'{score:.2f},{snr:.2f}']
    with open(f'./../../data/distance_estimation/music{music_type}_mono/pesq.csv', mode='a', newline='', encoding='utf-8') as file_pesq:
        writer = csv.writer(file_pesq)

        for entry in pesq_and_snr:
            dist_m, dist_mm = entry.split(',')
            writer.writerow([dist_m, dist_mm])

    pesq_scores.append(score)

    sf.write(f'./../../sound/distance_estimation/music{music_type}_mono/embded_music{music_type}_gain={amp:.2f}.wav', y1_emb, sampling_rate)

    # 確認用の表示
    # print(f'{(int(num+1) / loop_times)*100:3.0f}% Completed')

delay_time_errors = np.array(delay_time_errors)
pesq_scores = np.array(pesq_scores)

# ------------------------------
# 12th: 埋込強度の変化に伴う推定誤差結果の変化をプロット
# ------------------------------
fig = plt.figure(num='埋込強度変化', figsize=(6, 3))
plt.subplots_adjust(bottom=0.15)
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(embedding_amplitudes, delay_time_errors*speed_of_sound/1000, label='Distance Estimation Error')
ax1.set_xlim([-0.05,1.0])
ax1.set_xlabel("Embedding Amplitude Gain")
ax1.set_ylabel("Estimation Distance Error[m]")

ax2 = ax1.twinx()
ax2.plot(embedding_amplitudes, pesq_scores, 'r', label='PESQ')
ax2.set_ylim([-0.05, 5.5])
ax2.set_ylabel("PESQ")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc='lower right')

plt.savefig(f'./../../figure/distance_estimation/music{music_type}_Amp_vs_PESQ.png')
# plt.show()
