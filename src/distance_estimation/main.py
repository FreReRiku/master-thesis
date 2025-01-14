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
music_type      = 2
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
    # 10th: 遅延量を求める
    # ------------------------------

    # csp1_valuesとembedded_freq_weighted_csp_valuesに基づいて, 各遅延量(delay1, delay2)を推定し,
    # その結果をリスト delays に格納する.
    delays = []
    for csp1_signal, csp2_signal in zip(csp1_values, embedded_freq_weighted_csp_values):

        # CSP1における最初のピーク位置を取得
        csp1_peaks, _ = find_peaks(csp1_signal, height=0.5)
        first_csp1_peak_position = csp1_peaks[0]    # 最初に検出されたピーク位置

        # CSP2の最大値の位置を取得
        csp2_peak_position = np.argmax(csp2_signal)

        # 遅延量 (delay1, delay2) を配列として格納
        delay_pair = np.array([first_csp1_peak_position, csp2_peak_position])
        delays.append(delay_pair)

    # リストをnumpy配列に変換
    delays = np.array(delays)

    # 遅延推定誤差を計算する (平均絶対誤差)
    delay_errors = []
    for estimated_delay_pair in delays:
        # 推定された遅延量と基準値 (delay_adjusted_peak_positions) の差を計算
        error_direct = np.sum(np.abs(estimated_delay_pair - delay_adjusted_peak_positions))
        # 遅延推定ペアの順序が逆の場合の誤差を計算
        error_flipped = np.sum(np.abs(np.flip(estimated_delay_pair) - delay_adjusted_peak_positions))
        # 最小の誤差を選択してリストに追加
        delay_errors.append(np.min([error_direct, error_flipped]))
    # 遅延誤差の平均値を計算
    delay_error_mean = np.mean(np.array(delay_errors))
    # サンプル単位の誤差を時間単位 (ミリ秒) に変換
    mean_delay_error_ms = 1000 * (delay_error_mean / sampling_rate)
    # 結果をリストに保存
    delay_time_errors.append(mean_delay_error_ms)

    # --------------------
    # Peak Ratioを計算する
    # --------------------
    peak_ratios = []

    for csp2_signal, estimated_delay_pair in zip(embedded_freq_weighted_csp_values, delays):
        # CSP1が第1スピーカーと第2スピーカーどちらの遅延を検知したか判定
        # 判定結果を真の遅延位置 (true_delay_position) として保存
        if np.abs(estimated_delay_pair[0] - delay_adjusted_peak_positions[0]) < np.abs(estimated_delay_pair[0] - delay_adjusted_peak_positions[1]):
            true_delay_position = delay_adjusted_peak_positions[1]  # csp2はスピーカー2を推定したと判定
        else:
            true_delay_position = delay_adjusted_peak_positions[0]  # csp2はスピーカー1を推定したと判定

        # 真の遅延位置におけるピーク振幅を取得
        true_peak_amplitude = csp2_signal[true_delay_position]

        # 真の遅延位置以外での最大ピーク振幅を取得 (真の遅延位置をゼロにし, その次に大きい振幅を保存することで, 2番目に大きい振幅を得ている)
        csp2_signal_copy = np.copy(csp2_signal)
        csp2_signal_copy[true_delay_position] = 0   # 真の遅延位置をゼロに設定
        secondary_peak_amplitude = np.max(csp2_signal_copy)

        # Peak Ratioを計算し, リストに保存
        peak_ratios.append(true_peak_amplitude / (np.max([secondary_peak_amplitude, 1e-8])))

    # リストをnumpy配列に変換
    peak_ratios = np.array(peak_ratios)

    Peak_Ratio = [f'{np.mean(peak_ratios):.2f},{np.min(peak_ratios):.2f},{(peak_ratios[peak_ratios >= 1].size / peak_ratios.size)*100:2.0f}']
    with open(f'./../../data/distance_estimation/music{music_type}_mono/peak_ratio.csv', mode='a', newline='', encoding='utf-8') as file_peak_ratio:
        writer = csv.writer(file_peak_ratio)
        for entry in Peak_Ratio:
            dist_m, dist_mm, dist_mmm = entry.split(',')
            writer.writerow([dist_m, dist_mm, dist_mmm])

    # ------------------------------
    # 11th: 音質評価 (PESQとSNR)
    # ------------------------------

    # ISTFTを用いて時間波形に変換
    num_frames  = min([y1spec.shape[1], quality_check_y1spec.shape[1]])
    original_waveform = istft(y1spec[:,:num_frames], hop_length=hop_length)
    embedded_waveform = istft(quality_check_y1spec[:,:num_frames], hop_length=hop_length)

    # PESQ (音質スコア) の計算
    original_waveform_downsampled = resample(original_waveform[:sampling_rate * 5], orig_sr=sampling_rate, target_sr=16000)
    embedded_waveform_downsampled = resample(embedded_waveform[:sampling_rate * 5], orig_sr=sampling_rate, target_sr=16000)
    pesq_score = pesq(16000, original_waveform_downsampled, embedded_waveform_downsampled)

    # SNR (信号対雑音比) の計算
    signal_power = sum(original_waveform ** 2)
    noise_power  = sum((original_waveform - embedded_waveform) ** 2)
    snr = 20 * np.log10(signal_power / noise_power)

    pesq_and_snr = [f'{pesq_score:.2f},{snr:.2f}']
    with open(f'./../../data/distance_estimation/music{music_type}_mono/pesq.csv', mode='a', newline='', encoding='utf-8') as file_pesq:
        writer = csv.writer(file_pesq)

        for entry in pesq_and_snr:
            dist_m, dist_mm = entry.split(',')
            writer.writerow([dist_m, dist_mm])

    pesq_scores.append(pesq_score)

    sf.write(
        f'./../../sound/distance_estimation/music{music_type}_mono/embded_music{music_type}_gain={amplitude_gain:.2f}.wav',
        embedded_waveform,
        sampling_rate
    )

    # 確認用の表示
    # print(f'{(int(num+1) / loop_times)*100:3.0f}% Completed')

# numpy配列に変換
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
