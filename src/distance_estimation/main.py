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


for num, amplitude_gain in enumerate(embedding_amplitudes):
    # ------------------------------
    # オーディオファイルの読み込み
    # ------------------------------

    # ファイルパスの指定
    file_name_impulse1       = f'./../../sound/room_simulation/impulse_signal_ch1_{sampling_rate}Hz.wav'
    file_name_impulse2       = f'./../../sound/room_simulation/impulse_signal_ch2_{sampling_rate}Hz.wav'
    file_name_original       = f'./../../sound/original/music{music_type}_mono.wav'
    file_name_original_long  = f'./../../sound/original/long_music{music_type}_mono.wav'
    file_name_speaker1       = f'./../../sound/room_simulation/music{music_type}_room_ch1_{sampling_rate}Hz.wav'
    file_name_speaker2       = f'./../../sound/room_simulation/music{music_type}_room_ch2_{sampling_rate}Hz.wav'
    file_name_speaker_long   = f'./../../sound/room_simulation/long_music{music_type}_room_ch1_{sampling_rate}Hz.wav'
    # 読み込み
    impulse1, _ = sf.read(file_name_impulse1)
    impulse2, _ = sf.read(file_name_impulse2)
    x, _        = sf.read(file_name_original)
    xlong, _    = sf.read(file_name_original_long)
    y1, _       = sf.read(file_name_speaker1)
    y2, _       = sf.read(file_name_speaker2)
    ylong, _    = sf.read(file_name_speaker_long)

    # スペクトログラム
    xspec   = stft(x, n_fft=2*frame_length, hop_length=hop_length, win_length=frame_length, center=False)
    y1spec  = stft(y1, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)
    y2spec  = stft(y2, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)
    quality_check_y1spec  = stft(ylong, n_fft=2*frame_length, hop_length=hop_length, win_length=2*frame_length, center=False)

    # 保存用の配列
    csp0_values = []
    csp1_values = []
    csp2_values = []
    embedded_freq_csp1_values = []
    embedded_freq_csp2_values = []
    csp_difference_values     = []
    weighted_csp_difference_values      = []
    embedded_freq_csp_difference        = []
    embedded_freq_weighted_csp_values   = []

    # ------------------------------
    # 1st: CSP0, 及び 遅延距離d_0 の推定
    # ------------------------------
    for frame_start_index in pos_st_frame:

        # マイク入力音声のスペクトログラム(スペクトログラムの合成)
        yspec = y1spec + y2spec

        # --CSP0を求める(GCC-PHAT法)----------
        # クロススペクトラムの計算
        cross_spectrum_csp0 = xspec[:, frame_start_index : frame_start_index + num_embedding_frames] * np.conj(yspec[:, frame_start_index:frame_start_index + num_embedding_frames])
        # クロススペクトラムの振幅を計算
        cross_spectrum_magnitude_csp0 = np.abs(cross_spectrum_csp0)
        # 振幅がゼロに近い場合のゼロ除算を回避するための調整
        cross_spectrum_magnitude_csp0[cross_spectrum_magnitude_csp0  < epsilon] = epsilon
        # 白色化相互スペクトル (周波数領域)
        csp0_spectral = cross_spectrum_csp0 / cross_spectrum_magnitude_csp0
        # 周波数領域のCSP0を時間方向で平均
        csp0_average_spectral = np.mean(csp0_spectral, axis=1)
        # 周波数領域から時間領域へ変換 (逆STFT)
        csp0_time_domain = irfft(csp0_average_spectral, axis=0)

        # -CSP0の整形-----
        # 不要な後半部を除去
        csp0_time_domain = csp0_time_domain[:frame_length]
        # 最大値で正規化
        csp0_time_domain = csp0_time_domain / np.max(csp0_time_domain)

        # -CSP0の保存-----
        csp0_values.append(csp0_time_domain)

        # -dを推定-----
        # 注: 現在のオフセット値(67)を動的に調整する仕組みが必要
        estimated_delay = (np.argmax(csp0_values)-67)

    # --インパルスのピーク位置の推定----------
    # 最初に検出されたインパルス応答のピーク位置の記録用
    first_detected_peak_positions = []
    # 遅延時間を除いたインパルス応答ピークの記録用
    delay_adjusted_peak_positions = []

    # find_peaks関数を用いて、ピークを推定・記録
    for impulse_response in [impulse1, impulse2]:
        # find_peaks関数を使って, ピークを推定・記録
        peak_positions, _ = find_peaks(impulse_response, height=0.6)        # height: ピーク検出の閾値
        first_detected_peak_positions.append(peak_positions[0])                    # 最初のピークを記録
        delay_adjusted_peak_positions.append(peak_positions[0]-estimated_delay)    # 遅延時間を調整

    # numpy配列に変換
    first_detected_peak_positions = np.array(first_detected_peak_positions)
    delay_adjusted_peak_positions = np.array(delay_adjusted_peak_positions)

    # デバッグ用
    # print(first_detected_peak_positions)
    # print(delay_adjusted_peak_positions)

    # 遅延時間を考慮して音声信号をトリミング
    x_delay_adjusted_signal       = xlong[start_sample-estimated_delay : end_sample-estimated_delay]

    # 遅延時間を考慮したスペクトログラムを生成
    x_delay_adjusted_spectrogram  = stft(
        x_delay_adjusted_signal,
        n_fft       = 2 * frame_length,
        hop_length  = hop_length,
        win_length  = frame_length,
        center=False
    )

    for frame_start_index in pos_st_frame:

        # 埋め込み用の配列
        embedded_y1spec   = np.copy(y1spec)

        # マイク入力音声のスペクトログラム (スペクトログラムの合成)
        yspec       = y1spec + y2spec

        # ------------------------------
        # 2nd: CSP1を求める
        # ------------------------------
        # クロススペクトラムの計算
        cross_spectrum_csp1 = x_delay_adjusted_spectrogram[:, frame_start_index:frame_start_index+num_embedding_frames] * np.conj(yspec[:, frame_start_index:frame_start_index+num_embedding_frames])
        # クロススペクトラムの振幅を計算
        cross_spectrum_magnitude_csp1 = np.abs(cross_spectrum_csp1)
        # 振幅がゼロに近い場合のゼロ除算を回避するための調整
        cross_spectrum_magnitude_csp1[cross_spectrum_magnitude_csp1 < epsilon] = epsilon
        # 白色化相互スペクトラム (周波数領域)
        csp1_spectral = cross_spectrum_csp1 / cross_spectrum_magnitude_csp1
        # 周波数領域のCSP1を時間方向で平均
        csp1_average_spectral   = np.mean(csp1_spectral, axis=1)
        # 周波数領域から時間領域へ変換 (逆STFT)
        csp1_time_domain     = irfft(csp1_average_spectral, axis=0)
        # -CSP1の整形-----
        # 不要な後半部を除去
        csp1_time_domain     = csp1_time_domain[:frame_length]
        # 最大値で正規化
        csp1_time_domain     = csp1_time_domain / np.max(csp1_time_domain)

        # ------------------------------
        # 3rd: 埋め込み周波数のみのCSP1を求める
        # ------------------------------
        # --ゼロ埋め込み周波数の決定----------
        # 振幅(周波数?)の大きい順にインデックスを取得
        sorted_frequency_indices    = np.argsort(-np.abs(csp1_average_spectral))
        # CSPの最大embedding_frequency_bins個の周波数
        embedded_frequencies        = sorted_frequency_indices[:embedding_frequency_bins]

        # --埋め込み周波数のみのCSP1を求める----------
        # CSP1と同じ形状の配列を作成し, 全ての値を0に初期化
        csp1_embedded_spectral      = np.zeros_like(csp1_average_spectral)
        # 選択された周波数成分だけをコピー
        csp1_embedded_spectral[embedded_frequencies] = csp1_average_spectral[embedded_frequencies]
        # 特定の周波数成分だけを含む信号を時間領域に変換 (逆STFT)
        csp1_embedded_time_domain   = irfft(csp1_embedded_spectral, axis=0)

        # -埋め込み周波数のみのCSP1の整形-----
        # 不要な後半部を削除
        csp1_embedded_time_domain   = csp1_embedded_time_domain[:frame_length]
        # 最大値で正規化
        csp1_embedded_time_domain   = csp1_embedded_time_domain / np.max(csp1_embedded_time_domain)

        # ------------------------------
        # 4th: 振幅変調と位相変調
        # ------------------------------
        # 振幅変調: 選択された周波数成分に対して振幅を変更
        embedded_y1spec[embedded_frequencies, :] = amplitude_gain * embedded_y1spec[embedded_frequencies, :]        # embedded_freqの周波数ビンにamp倍
        # 位相変調: 選択された周波数成分に対して位相を変更
        phase_shift = embedding_phase / 180 * np.pi
        embedded_y1spec[embedded_frequencies, :] = embedded_y1spec[embedded_frequencies, :] * np.exp(1j * phase_shift)
        # print(f'Y1emb shape: {y1emb.shape}')

        # 音質検査用の振幅変調 (埋め込み用の配列に適用)
        quality_check_y1spec[embedded_frequencies, frame_start_index:frame_start_index+3] = amplitude_gain * quality_check_y1spec[embedded_frequencies, frame_start_index:frame_start_index+3]

        # ------------------------------
        # 5th: CSP2を求める
        # ------------------------------
        # 埋め込み信号を利用している(embedded_y1spec)
        yspec   = embedded_y1spec + y2spec
        # 相互スペクトラム (クロススペクトラム) の計算
        cross_spectrum_csp2 = xspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames] * np.conj(yspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames])
        # クロススペクトラムの振幅を計算
        cross_spectrum_magnitude_csp2 = np.abs(cross_spectrum_csp2)
        # 振幅がゼロに近い場合のゼロ除算を回避するための調整
        cross_spectrum_magnitude_csp2[cross_spectrum_magnitude_csp2 < epsilon] = epsilon
        # 白色化相互スペクトラム (周波数領域)
        csp2_spectral = cross_spectrum_csp2 / cross_spectrum_magnitude_csp2
        # 周波数領域のCSP2を時間方向で平均
        csp2_average_spectral = np.mean(csp2_spectral, axis=1)
        # 周波数領域から時間領域へ変換 (逆STFT)
        csp2_time_domain      = irfft(csp2_average_spectral, axis=0)

        # -CSP2の整形-----
        # いらない後半部を除去
        csp2_time_domain     = csp2_time_domain[:frame_length]
        # 最大で割り算
        csp2_time_domain     = csp2_time_domain / np.max(csp2_time_domain)


        # ------------------------------
        # 6th: 埋め込み周波数のみのCSP2を求める
        # ------------------------------
        # csp2_average_spectralと同じ形状の配列を作成し, 全ての値を0に初期化
        csp2_embedded_spectral = np.zeros_like(csp2_average_spectral)
        # 選択された周波数成分 (embedded_frequencies) のみをコピー
        csp2_embedded_spectral[embedded_frequencies] = csp2_average_spectral[embedded_frequencies]
        # 特定の周波数成分だけを含む信号を時間領域に変換する
        csp2_embedded_time_domain = irfft(csp2_embedded_spectral, axis=0)

        # -CSP2(埋込周波数のみ)の整形-----
        # 不要な後半部を除去
        csp2_embedded_time_domain = csp2_embedded_time_domain[:frame_length]
        # 最大値で正規化
        csp2_embedded_time_domain = csp2_embedded_time_domain / np.max(csp2_embedded_time_domain)

        # ------------------------------
        # 7th: 重み付き差分CSPを求める
        # ------------------------------
        # -重みを計算する-----
        # CSP1のピーク位置を計算
        csp1_peak_positions, _ = find_peaks(csp1_time_domain, threshold=0.01)
        # ピーク位置をピークの大きい順にソート
        sorted_peak_indices    = np.argsort(-csp1_time_domain[csp1_peak_positions])
        # 最大embedding_frequency_bins個のピーク位置を取得
        selected_peak_positions = csp1_peak_positions[sorted_peak_indices[:embedding_frequency_bins]]
        # 第1スピーカーの遅延推定 (最大ピーク位置)
        primary_speaker_delay = selected_peak_positions[0]

        # -重みの計算-----
        csp1_weights      = np.copy(csp1_time_domain)
        # 推定した第1スピーカーのピーク付近の値を0に設定
        csp1_weights[primary_speaker_delay - 3: primary_speaker_delay + 3] = 0
        # 閾値以下の値を0に設定
        csp1_weights[csp1_weights < threshold_ratio] = 0
        # 重みを正規化
        csp1_weights = csp1_weights / np.max(np.abs(csp1_weights))

        # ------------------------------
        # 8th: 重み付け差分CSPによる遅延推定
        # ------------------------------
        # CSPの差分
        csp_difference = csp1_time_domain - csp2_time_domain
        # 差分CSPを正規化
        normalized_csp_difference = csp_difference / np.max(csp_difference)

        # 重み付け差分CSP
        weighted_csp_difference = csp1_weights * normalized_csp_difference

        # ------------------------------
        # 9th: 重み付け差分CSP(埋込周波数のみ)用の重み計算
        # ------------------------------
        # 埋め込み周波数成分を含むCSP1のピーク位置を計算
        embedded_csp1_peak_positions, _ = find_peaks(csp1_embedded_time_domain, threshold=0.01)
        # ピーク位置をピークの大きい順にソート
        sorted_embedded_peak_indices = np.argsort(-csp1_embedded_time_domain[embedded_csp1_peak_positions])
        # 最大embedding_frequency_bins個のピーク位置を取得
        selected_embedded_peak_positions = embedded_csp1_peak_positions[sorted_embedded_peak_indices[:embedding_frequency_bins]]
        # 第1スピーカーの遅延推定 (最大ピーク位置)
        primary_embedded_speaker_delay = selected_embedded_peak_positions[0]

        # 重みの計算
        embedded_csp1_weights = np.copy(csp1_embedded_time_domain)
        # 推定した第1スピーカーのピーク付近の値を0に設定
        embedded_csp1_weights[primary_embedded_speaker_delay - 3 : primary_embedded_speaker_delay + 3] = 0
        # 閾値以下の値を0に設定
        embedded_csp1_weights[embedded_csp1_weights < threshold_ratio] = 0
        # 重みを正規化
        embedded_csp1_weights = embedded_csp1_weights / np.max(np.abs(embedded_csp1_weights))

        # ------------------------------
        # 10th: 重み付け差分CSP(埋込周波数のみ)による遅延推定
        # ------------------------------
        # 埋め込み周波数におけるCSPの差分
        embedded_csp_difference = csp1_embedded_time_domain - csp2_embedded_time_domain
        # 差分CSPを正規化
        normalized_embedded_csp_difference = embedded_csp_difference / np.max(embedded_csp_difference)

        # 重み付け埋込差分CSP
        weighted_embedded_csp_difference = csp1_weights * normalized_embedded_csp_difference

        # ------------------------------
        # 9th: 計算結果を保存する
        # ------------------------------
        csp1_values.append(csp1_time_domain)                                        # CSP1
        csp2_values.append(csp2_time_domain)                                        # CSP2
        embedded_freq_csp1_values.append(csp1_embedded_time_domain)                 # 特定の周波数成分だけを抽出したCSP1
        embedded_freq_csp2_values.append(csp2_embedded_time_domain)                 # 特定の周波数成分だけを抽出したCSP2
        csp_difference_values.append(normalized_csp_difference)                     # 差分CSP
        weighted_csp_difference_values.append(weighted_csp_difference)              # 重み付け差分CSP
        embedded_freq_csp_difference.append(normalized_embedded_csp_difference)     # 特定の周波数成分だけを抽出した差分CSP
        embedded_freq_weighted_csp_values.append(weighted_embedded_csp_difference)  # 特定の周波数成分だけを抽出した重み付け差分CSP

    # numpyに変更
    csp1_values                         = np.array(csp1_values)
    csp2_values                         = np.array(csp2_values)
    embedded_freq_csp1_values           = np.array(embedded_freq_csp1_values)
    embedded_freq_csp2_values           = np.array(embedded_freq_csp2_values)
    csp_difference_values               = np.array(csp_difference_values)
    weighted_csp_difference_values      = np.array(weighted_csp_difference_values)
    embedded_freq_csp_difference        = np.array(embedded_freq_csp_difference)
    embedded_freq_weighted_csp_values   = np.array(embedded_freq_weighted_csp_values)

    # 推定誤差を算出
    distance_speaker1 = [f'{first_detected_peak_positions[0]/sampling_rate*speed_of_sound:.2f},{1000*first_detected_peak_positions[0]/sampling_rate:.2f}']
    distance_speaker2 = [f'{first_detected_peak_positions[1]/sampling_rate*speed_of_sound:.2f},{1000*first_detected_peak_positions[1]/sampling_rate:.2f}']
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

    # csp1_valuesとembedded_freq_weighted_csp_valuesに基づいて, 遅延量(delay)を推定し,
    # その結果をリストDelayに格納する.
    Delay = []
    for csp1, csp2 in zip(csp1_values, embedded_freq_weighted_csp_values):
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
        # delay(推定遅延量)とdelay_adjusted_peak_positions(基準値)の差の絶対値を計算し, それらを足し合わせる.
        tmp1 = np.sum(np.abs(delay - delay_adjusted_peak_positions))
        # 遅延ペア([delay1, delay2])の順序が逆である場合の誤差を計算.
        # 遅延ペアの順序が異なっている場合の比較を考慮している.
        tmp2 = np.sum(np.abs(np.flip(delay) - delay_adjusted_peak_positions))
        # tmp1, tmp2のうち, 小さい方の値(最小誤差)を選択して, リストerrorに追加.
        error.append(np.min([tmp1, tmp2]))
    # リストerrorに格納されたすべての遅延誤差の平均を計算し,
    # 全体としての平均的な遅延誤差を取得.
    error = np.mean(np.array(error))
    # サンプル数単位から時間単位へ変換
    mean_delay_error_ms = 1000 * (error / sampling_rate)
    delay_time_errors.append(mean_delay_error_ms)

    PR_data = []
    for csp2, delay in zip(embedded_freq_weighted_csp_values, Delay):
        # まずcsp1が第１スピーカと第２スピーカどちらの遅延を検知したか判定
        # 結果をpos_truthに保存.
        if np.abs(delay[0] - delay_adjusted_peak_positions[0]) < np.abs(delay[0] - delay_adjusted_peak_positions[1]):
            pos_truth = delay_adjusted_peak_positions[1]  # csp2はpos_imp[1]を推定したと判定
        else:
            pos_truth = delay_adjusted_peak_positions[0]  # csp2はpos_imp[0]を推定したと判定

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
    frames  = min([y1spec.shape[1], quality_check_y1spec.shape[1]])
    y1_orig = istft(y1spec[:,:frames], hop_length=hop_length)
    y1_emb  = istft(quality_check_y1spec[:,:frames], hop_length=hop_length)

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

    sf.write(f'./../../sound/distance_estimation/music{music_type}_mono/embded_music{music_type}_gain={amplitude_gain:.2f}.wav', y1_emb, sampling_rate)

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
