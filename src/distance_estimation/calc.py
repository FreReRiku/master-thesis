'''
calc.py
----------

GCC-PHAT法を用いて相互相関, 並びに遅延推定を求めます.

Created by FreReRiku on 2025/01/17
'''

import numpy as np
import soundfile as sf
import save
import csv
from pesq import pesq
from librosa import stft, istft, resample
from scipy.signal import find_peaks
from scipy.fft import irfft

def gcc_phat(music_type):

    # ------------------------------
    # 1. パラメータの設定
    # ------------------------------

    # 音源タイプの指定
    music_type = music_type

    # 極小値をゼロ除算防止のために使用
    epsilon = 1e-20

    # 音速 [m/s]
    sound_of_speed = 340.29

    # 音響データのサンプリング周波数 [Hz]
    sampling_rate   = 44100

    # シミュレーションで使用する信号のサンプル数 [サンプル数]
    signal_length_samples = 44100 * 5   # 5 秒間の信号を仮定

    # FFT点数
    n_fft = 2048

    # STFTのフレームサイズ [サンプル数]
    frame_length    = 1024      # フレームあたりのデータポイント数

    # STFTのホップサイズ [サンプル数]
    hop_length      = 256       # 隣接フレーム間の重なり

    # STFTのウィンドウサイズ
    win_length = 1024

    # 信号をトリミングする開始点と終了点 [サンプル数]
    start_sample    = 1000      # トリミングの開始位置
    end_sample      = start_sample + signal_length_samples # 信号の終了位置

    # フレーム開始位置をずらす回数
    num_trials      = 100       # フレーム開始位置を 100 回シフトする

    # 埋め込みを行うフレーム数 [フレーム数] (論文では M と表記)
    num_embedding_frames        = 40    # 連続して特定の振幅を埋め込むフレーム数

    # 埋め込みを行う周波数の範囲 (STFTフレーム内のビン数, 論文中では L と表記)
    embedding_frequency_bins    = np.floor(frame_length*0.1).astype(int)    # フレーム長の10%を埋め込み対象にしている

    # 各試行におけるフレーム開始位置
    # フレーム開始位置のシフトリストを作成
    # 3フレームずつずらしながら, num_shift_count 回分を生成
    pos_st_frame = np.arange(0, num_trials*3, 3)

    # frame_count = round(16000*3/16000)     # フレームカウント
    # pos_st_frame = np.arange(0, num_trials*frame_count, frame_count)

    # CSPの最大値に対するノイズと判定する振幅比のしきい値(const)
    threshold_ratio  = 0.3  # ピーク振幅がこの値以下の場合ノイズと判定

    # --------------------------
    # 2. 設定したパラメーターを記録
    # --------------------------

    # ----------------
    # 3. 埋め込み設定
    # ----------------

    # 埋め込み振幅の設定
    # 埋め込み処理の試行回数を定義 (振幅を段階的に変化させてテスト)
    loop_times = 25     # 振幅を25段階に設定

    # 埋め込み振幅の値 (0 から 1 まで均等に分割した値, 論文中のNを0から1に変化させている)
    embedding_amplitudes    = np.linspace(0, 1, loop_times)

    # 埋め込む位相の設定 (現在未使用)
    # embedding_phase  = 0  # 必要に応じて位相変化を追加可能

    # --------------------
    # 4. データ格納用のリスト
    # --------------------

    # 遅延推定誤差を記録するリスト
    delay_time_errors = []  # 指定された遅延量と真値の誤差を格納

    # 音質評価 (PESQスコア) を記録するリスト
    pesq_scores = []        # 埋め込み処理後の音質評価スコア

    for num, amplitude_gain in enumerate(embedding_amplitudes):

        # ------------------------------
        # 5. オーディオファイルの読み込み
        # ------------------------------

        # ファイルパスの指定
        file_name_impulse1       = f'./../../sound/room_simulation/impulse_signal_ch1_{sampling_rate}Hz.wav'
        file_name_impulse2       = f'./../../sound/room_simulation/impulse_signal_ch2_{sampling_rate}Hz.wav'
        file_name_original       = f'./../../sound/original/music{music_type}_mono.wav'
        file_name_speaker1       = f'./../../sound/room_simulation/music{music_type}_room_ch1_{sampling_rate}Hz.wav'
        file_name_speaker2       = f'./../../sound/room_simulation/music{music_type}_room_ch2_{sampling_rate}Hz.wav'

        # オーディオデータをファイルから読み込み
        impulse1, _ = sf.read(file_name_impulse1)
        impulse2, _ = sf.read(file_name_impulse2)
        x, _        = sf.read(file_name_original)
        y1, _       = sf.read(file_name_speaker1)
        y2, _       = sf.read(file_name_speaker2)

        # ------------------------------
        # 6. オーディオファイルの編集
        # ------------------------------

        # インパルス応答 (チャンネル1と2の合成) を生成
        impulse = impulse1[:2500] + impulse2[:2500]  # 各インパルス応答を2500サンプルでトリミング後, 足し合わせ

        # 各オーディオ信号を指定した範囲でトリミング
        x_0     = x[start_sample:end_sample]  # オリジナル音源のトリミング範囲
        y1_0    = y1[start_sample:end_sample]  # スピーカーS1から収録した音のトリミング範囲
        y2_0    = y2[start_sample:end_sample]  # スピーカーS2から収録した音のトリミング範囲

        # STFT (短時間フーリエ変換) によりスペクトログラムを生成
        xspec       = stft(x_0,  n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
        y1spec      = stft(y1_0, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
        y2spec      = stft(y2_0, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
        y1zero      = stft(y1,   n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)

        # ------------------------------
        # 7. データ格納用のリストの初期化
        # ------------------------------

        # 各スピーカーとの距離・到来時間に関するデータを記録するリスト
        distance_speaker1 = []  # スピーカーS1とマイク間の距離・到来時間の記録用
        distance_speaker2 = []  # スピーカーS2とマイク間の距離・到来時間の記録用

        # 埋め込まれた周波数に関する情報を記録するリスト
        store_embedded_frequencies = [] # 埋め込まれた周波数を記録

        # 各CSP (Cross-power Spectrum Phase) の計算結果を記録するリスト
        csp0_values = []    # 初期のCSP (基準)
        csp1_values = []    # スピーカーS1に基づくCSP
        csp2_values = []    # スピーカーS2に基づくCSP

        # 埋め込み周波数成分に基づくCSPの計算結果を記録するリスト
        embedded_freq_csp1_values = []  # スピーカーS1 (埋め込み周波数のみ)
        embedded_freq_csp2_values = []  # スピーカーS2 (埋め込み周波数のみ)

        # CSP間の差分に基づくデータを記録するリスト
        csp_difference_values = []              # CSPの差分
        weighted_csp_difference_values = []     # 重み付けされたCSPの差分
        embedded_freq_csp_difference = []       # 埋め込み周波数のみのCSP差分
        embedded_freq_weighted_csp_values = []  # 埋め込み周波数のみの重み付けCSP


        # ------------------------------
        # 8. CSP0, d_0 の推定
        # ------------------------------
        for frame_start_index in pos_st_frame:

            # マイク入力音声のスペクトログラムを合成
            # - スピーカーS1及びS2からの信号を足し合わせたスペクトログラム
            yspec = y1spec + y2spec

            # --CSP0を求める(GCC-PHAT法)----------
            # クロススペクトラムの計算
            # - マイク信号と音源信号の周波数領域での相関を計算
            cross_spectrum_csp0 = yspec[:, frame_start_index:frame_start_index+num_embedding_frames] * np.conj(xspec[:, frame_start_index:frame_start_index+num_embedding_frames])

            # クロススペクトラムの振幅を計算
            # - 周波数ごとのエネルギーの絶対値を取得
            cross_spectrum_magnitude_csp0 = np.abs(cross_spectrum_csp0)

            # 振幅がゼロに近い場合のゼロ除算を回避
            # - 振幅がepsilon未満の場合にepsilonで置き換え
            cross_spectrum_magnitude_csp0[cross_spectrum_magnitude_csp0  < epsilon] = epsilon

            # 白色化相互スペクトル (CSP) を計算
            # - クロススペクトラムを振幅で正規化
            csp0_spectral = cross_spectrum_csp0 / cross_spectrum_magnitude_csp0

            # 周波数領域のCSP0を時間方向で平均
            # - 各フレームのスペクトルを時間方向に統合
            csp0_average_spectral = np.mean(csp0_spectral, axis=1)

            # 周波数領域から時間領域へ変換 (逆STFT)
            # - 時間領域信号に変換することで到来時間情報を取得
            csp0_time_domain = irfft(csp0_average_spectral, axis=0)

            # -CSP0の整形-----
            # 不要な後半部を除去 (フレーム長に合わせてトリミング)
            csp0_time_domain = csp0_time_domain[:frame_length]

            # 最大値で正規化
            # - 最大値を1とするスケール調整
            csp0_time_domain = csp0_time_domain / np.max(csp0_time_domain)

            # -CSP0の保存-----
            # - 後の処理で利用するためリストに保存
            csp0_values.append(csp0_time_domain)

            # -d_0 を推定-----
            # 最大値の位置 (遅延時間) を推定
            # 注: 現在のオフセット値(25)を動的に調整する仕組みが必要
            estimated_delay = (np.argmax(csp0_values)-25)
        
        # ------------------------------
        # 9. インパルスのピーク位置の推定
        # ------------------------------

        # 初期化: インパルス応答のピーク位置を記録するリスト
        first_detected_peak_positions = []      # 最初に検出されたピーク位置
        delay_adjusted_peak_positions = []      # 推定遅延を調整後のピーク位置

        # インパルス応答におけるピークの検出
        for impulse_response in [impulse1, impulse2]:
            # ピーク位置を検出 (高さの閾値: 0.6)
            peak_positions, _ = find_peaks(impulse_response, height=0.6)

            # 最初のピーク位置を記録
            first_detected_peak_positions.append(peak_positions[0])

            # 推定された遅延を調整後のピーク位置を記録
            delay_adjusted_peak_positions.append(peak_positions[0]-estimated_delay)

        # ------------------------------
        # 10. 遅延時間を考慮して信号をトリミング
        # ------------------------------

        # 推定遅延時間 (d_0) を考慮してオリジナル音源をトリミング
        x = x[start_sample-estimated_delay : end_sample-estimated_delay]

        # トリミング後の音源を基にスペクトログラムを再生成
        # - 遅延時間を考慮した状態でSTFTを計算
        xspec = stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)

        for frame_start_index in pos_st_frame:

            # 埋め込み用の配列を初期化
            # - スピーカーS1のスペクトログラムをコピーして埋め込み操作用に利用
            embedded_y1spec   = np.copy(y1spec)

            # マイク入力音声のスペクトログラムを合成
            # - スピーカーS1とS2の信号を足し合わせたスペクトログラム
            yspec       = y1spec + y2spec

            # ------------------------------
            # 11. CSP1を求める
            # ------------------------------
            # クロススペクトラムを計算
            # - スピーカーの合成信号とオリジナル音源の周波数領域での相関
            cross_spectrum_csp1 = yspec[:, frame_start_index:frame_start_index+num_embedding_frames] * np.conj(xspec[:, frame_start_index:frame_start_index+num_embedding_frames])

            # クロススペクトラムの振幅を計算
            # - 各周波数成分のエネルギーの絶対値を取得
            cross_spectrum_magnitude_csp1 = np.abs(cross_spectrum_csp1)

            # 振幅がゼロに近い場合のゼロ除算を回避するための調整
            # - 振幅がepsilon未満の場合にepsilonで置き換え
            cross_spectrum_magnitude_csp1[cross_spectrum_magnitude_csp1 < epsilon] = epsilon

            # 白色化相互スペクトラム (周波数領域)
            # - クロススペクトラムを振幅で正規化
            csp1_spectral = cross_spectrum_csp1 / cross_spectrum_magnitude_csp1

            # 周波数領域のCSP1を時間方向で平均
            # - 各フレームのスペクトルを時間方向に統合
            csp1_average_spectral   = np.mean(csp1_spectral, axis=1)

            # 周波数領域から時間領域へ変換 (逆STFT)
            # - 時間領域信号に変換することで到来時間情報を取得
            csp1_time_domain     = irfft(csp1_average_spectral, axis=0)

            # -CSP1の整形-----
            # 不要な後半部を除去 (フレーム長に合わせてトリミング)
            csp1_time_domain     = csp1_time_domain[:frame_length]

            # 最大値で正規化
            # - 最大値を1とするスケール調整
            csp1_time_domain     = csp1_time_domain / np.max(csp1_time_domain)

            # --------------------------------
            # 12. 埋め込み周波数のみのCSP1を求める
            # --------------------------------

            # --ゼロ埋め込み周波数の決定----------
            # 振幅の大きい順に周波数インデックスを取得
            sorted_frequency_indices    = np.argsort(-np.abs(csp1_average_spectral))

            # CSPの最大embedding_frequency_bins個の周波数を選択
            embedded_frequencies        = sorted_frequency_indices[:embedding_frequency_bins]

            # --埋め込み周波数のみのCSP1を求める----------
            # CSP1と同じ形状の配列を初期化 (全て0)
            csp1_embedded_spectral      = np.zeros_like(csp1_average_spectral)

            # 選択された周波数成分のみをコピー
            #振幅が大きい周波数成分（埋め込み対象周波数）を embedded_frequencies にリストとして取得
            #その周波数成分のみを csp1_embedded_spectral にコピーし、他の成分はゼロのままにする
            #結果として、CSP1の中でも特に振幅が大きい周波数成分だけを残したスペクトルを作る
            csp1_embedded_spectral[embedded_frequencies] = csp1_average_spectral[embedded_frequencies]

            # 特定の周波数成分だけを含む信号を時間領域に変換 (逆STFT)
            csp1_embedded_time_domain   = irfft(csp1_embedded_spectral, axis=0)

            # -埋め込み周波数のみのCSP1の整形-----
            # 不要な後半部を削除 (フレーム長に合わせてトリミング)
            csp1_embedded_time_domain   = csp1_embedded_time_domain[:frame_length]

            # 最大値で正規化
            # - 最大値を1とするスケール調整
            csp1_embedded_time_domain   = csp1_embedded_time_domain / np.max(csp1_embedded_time_domain)

            # ------------------------------
            # 13. 振幅変調と位相変調
            # ------------------------------
            # 振幅変調: 選択された周波数成分に対して振幅を変更
            # - 埋め込み対象の周波数ビンに指定した振幅ゲイン (amplitude_gain) を適用
            embedded_y1spec[embedded_frequencies, :] = amplitude_gain * embedded_y1spec[embedded_frequencies, :]

            # 位相変調: 選択された周波数成分の位相を調整 (現在は無効)
            # - 位相変調を適用する場合はコメントを解除してください
            # phase_shift = embedding_phase / 180 * np.pi       # 度単位の位相シフトをラジアンに変換
            # embedded_y1spec[embedded_frequencies, :] = embedded_y1spec[embedded_frequencies, :] * np.exp(1j * phase_shift)

            # 音質検査用の振幅変調
            # - 埋め込み対象の配列 (y1zero) に対して振幅ゲインを適用
            y1zero[embedded_frequencies, frame_start_index:frame_start_index+3] = amplitude_gain * y1zero[embedded_frequencies, frame_start_index:frame_start_index+3]

            # ------------------------------
            # 14. CSP2を求める
            # ------------------------------
            # 埋め込み信号を利用
            # - スピーカーS1の埋め込みスペクトログラムとスピーカーS2のスペクトログラムを作成
            yspec   = embedded_y1spec + y2spec

            # クロススペクトラムを計算
            # - 埋め込み信号と音源信号の周波数領域での相関を取得
            cross_spectrum_csp2 = yspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames] * np.conj(xspec[:, frame_start_index+num_embedding_frames:frame_start_index+2*num_embedding_frames])
            
            # クロススペクトラムの振幅を計算
            # - 各周波数成分のエネルギーの絶対値を取得
            cross_spectrum_magnitude_csp2 = np.abs(cross_spectrum_csp2)

            # 振幅がゼロに近い場合のゼロ除算を回避するための調整
            # - 振幅がepsilon未満の場合にepsilonで置き換え
            cross_spectrum_magnitude_csp2[cross_spectrum_magnitude_csp2 < epsilon] = epsilon

            # 白色化相互スペクトラム (CSP2) を計算
            # - クロススペクトラムを振幅で正規化
            csp2_spectral = cross_spectrum_csp2 / cross_spectrum_magnitude_csp2

            # 周波数領域のCSP2を時間方向で平均
            # - 各フレームのスペクトルを時間方向に統合
            csp2_average_spectral = np.mean(csp2_spectral, axis=1)

            # 周波数領域から時間領域へ変換 (逆STFT)
            # - 時間領域信号に変換することで到来時間情報を取得
            csp2_time_domain      = irfft(csp2_average_spectral, axis=0)

            # -CSP2の整形-----
            # 不要な後半部を除去 (フレーム長に合わせてトリミング)
            csp2_time_domain     = csp2_time_domain[:frame_length]

            # 最大値で正規化
            # - 最大値を1とするスケール調整
            csp2_time_domain     = csp2_time_domain / np.max(csp2_time_domain)


            # --------------------------------
            # 15. 埋め込み周波数のみのCSP2を求める
            # --------------------------------

            # CSP2の埋め込み用スペクトルを初期化 (全て0)
            # - csp2_average_spectral と同じ形状
            csp2_embedded_spectral = np.zeros_like(csp2_average_spectral)

            # 選択された周波数成分のみをコピー
            # - 埋め込み周波数 (embedded_frequencies) の値を抽出
            csp2_embedded_spectral[embedded_frequencies] = csp2_average_spectral[embedded_frequencies]

            # 埋め込み周波数のみの信号を時間領域に変換 (逆STFT)
            csp2_embedded_time_domain = irfft(csp2_embedded_spectral, axis=0)

            # -CSP2(埋込周波数のみ)の整形-----
            # 不要な後半部を除去 (フレーム長に合わせてトリミング)
            csp2_embedded_time_domain = csp2_embedded_time_domain[:frame_length]
            # 最大値で正規化
            # - 最大値を1とするスケール調整
            csp2_embedded_time_domain = csp2_embedded_time_domain / np.max(csp2_embedded_time_domain)

            # ------------------------------
            # 16. 重み付き差分CSPを求める
            # ------------------------------

            # -重みを計算する-----
            # CSP1のピーク位置を計算
            # - 信号内で特に目立つピーク位置を検出
            csp1_peak_positions, _ = find_peaks(csp1_time_domain, threshold=0.01)

            # ピーク位置をピークの大きい順にソート
            # - 検出されたピークを振幅が高い順に並べ替え
            sorted_peak_indices    = np.argsort(-csp1_time_domain[csp1_peak_positions])

            # 最大embedding_frequency_bins個のピーク位置を取得
            # - 埋め込み対象とするピークを指定した数だけ選択
            selected_peak_positions = csp1_peak_positions[sorted_peak_indices[:embedding_frequency_bins]]

            # 第1スピーカーの遅延推定 (最大ピーク位置)
            # - 最も大きなピークを第1スピーカーの遅延として仮定
            primary_speaker_delay = selected_peak_positions[0]

            # -重みの計算-----
            # 重み配列を初期化 (CSP1のコピー)
            csp1_weights      = np.copy(csp1_time_domain)

            # 推定した第1スピーカーのピーク付近の値を0に設定
            # - 第1スピーカーのピークの影響を除去
            csp1_weights[primary_speaker_delay - 3: primary_speaker_delay + 3] = 0

            # 閾値以下の値を0に設定
            # - ノイズや小さなピークを除去
            csp1_weights[csp1_weights < threshold_ratio] = 0

            # 重みを正規化
            # - 最大値を1とするようスケールを調整
            csp1_weights = csp1_weights / np.max(np.abs(csp1_weights))

            # ------------------------------
            # 17. 重み付け差分CSPによる遅延推定
            # ------------------------------

            # CSPの差分
            # - CSP1 (スピーカーS1) と CSP2 (スピーカーS2) の差分を計算
            csp_difference = csp1_time_domain - csp2_time_domain

            # 差分CSPを正規化
            # - 最大値を1とするスケール調整
            normalized_csp_difference = csp_difference / np.max(csp_difference)

            # 重み付け差分CSP
            # - 重みを適用してスピーカーS1の影響を強調
            weighted_csp_difference = csp1_weights * normalized_csp_difference

            # ------------------------------
            # 18. 重み付け差分CSP(埋込周波数のみ)用の重み計算
            # ------------------------------

            # 埋め込み周波数成分を含むCSP1のピーク位置を計算
            # - CSP1 の埋め込み対象周波数に特化したピーク検出
            embedded_csp1_peak_positions, _ = find_peaks(csp1_embedded_time_domain, threshold=0.01)

            # ピーク位置をピークの大きい順にソート
            # - 検出されたピークを振幅が高い順に並べ替え
            sorted_embedded_peak_indices = np.argsort(-csp1_embedded_time_domain[embedded_csp1_peak_positions])

            # 最大embedding_frequency_bins個のピーク位置を取得
            # - 埋め込み対象とするピークを指定した数だけ選択
            selected_embedded_peak_positions = embedded_csp1_peak_positions[sorted_embedded_peak_indices[:embedding_frequency_bins]]

            # 第1スピーカーの遅延推定 (最大ピーク位置)
            # - 埋め込み周波数に限定して最大のピークを遅延として仮定
            primary_embedded_speaker_delay = selected_embedded_peak_positions[0]

            # 重みの計算
            # - 埋め込み周波数成分の重みを初期化 (CSP1の埋め込み版)
            embedded_csp1_weights = np.copy(csp1_embedded_time_domain)

            # 推定した第1スピーカーのピーク付近の値を0に設定
            embedded_csp1_weights[primary_embedded_speaker_delay - 3 : primary_embedded_speaker_delay + 3] = 0

            # 閾値以下の値を0に設定
            # - ノイズや小さなピークを除去
            embedded_csp1_weights[embedded_csp1_weights < threshold_ratio] = 0
            # 重みを正規化
            # - 最大値を1とするスケール調整
            embedded_csp1_weights = embedded_csp1_weights / np.max(np.abs(embedded_csp1_weights))

            # --------------------------------------------
            # 19. 重み付け差分CSP(埋込周波数のみ)による遅延推定
            # --------------------------------------------

            # 埋め込み周波数におけるCSPの差分
            # - 埋め込み対象周波数のCSP1とCSP2の差分を計算
            embedded_csp_difference = csp1_embedded_time_domain - csp2_embedded_time_domain

            # 差分CSPを正規化
            # - 最大値を1とするスケール調整
            normalized_embedded_csp_difference = embedded_csp_difference / np.max(embedded_csp_difference)

            # 重み付け埋込差分CSP
            # - 重みを適用して主スピーカーの影響を強調
            weighted_embedded_csp_difference = csp1_weights * normalized_embedded_csp_difference

            # ------------------------------
            # 20. 計算結果を保存する
            # ------------------------------
            # 計算された各種データをリストに追加
            # - CSP1とCSP2の時間領域信号, 埋め込み対象周波数, 差分CSP, 重み付け差分CSPなど
            csp1_values.append(csp1_time_domain)                                        # CSP1
            csp2_values.append(csp2_time_domain)                                        # CSP2
            store_embedded_frequencies.append(embedded_frequencies)                     # 埋め込み対象周波数
            embedded_freq_csp1_values.append(csp1_embedded_time_domain)                 # 特定の周波数成分だけを抽出したCSP1
            embedded_freq_csp2_values.append(csp2_embedded_time_domain)                 # 特定の周波数成分だけを抽出したCSP2
            csp_difference_values.append(normalized_csp_difference)                     # 差分CSP
            weighted_csp_difference_values.append(weighted_csp_difference)              # 重み付け差分CSP
            embedded_freq_csp_difference.append(normalized_embedded_csp_difference)     # 特定の周波数成分だけを抽出した差分CSP
            embedded_freq_weighted_csp_values.append(weighted_embedded_csp_difference)  # 特定の周波数成分だけを抽出した重み付け差分CSP

        # numpyに変更
        # - リスト形式からnumpy配列に変換し, 後続の処理で扱いやすくする
        first_detected_peak_positions       = np.array(first_detected_peak_positions)
        delay_adjusted_peak_positions       = np.array(delay_adjusted_peak_positions)
        store_embedded_frequencies          = np.array(store_embedded_frequencies)
        csp1_values                         = np.array(csp1_values)
        csp2_values                         = np.array(csp2_values)
        embedded_freq_csp1_values           = np.array(embedded_freq_csp1_values)
        embedded_freq_csp2_values           = np.array(embedded_freq_csp2_values)
        csp_difference_values               = np.array(csp_difference_values)
        weighted_csp_difference_values      = np.array(weighted_csp_difference_values)
        embedded_freq_csp_difference        = np.array(embedded_freq_csp_difference)
        embedded_freq_weighted_csp_values   = np.array(embedded_freq_weighted_csp_values)

        # ------------------------------
        # 21. 遅延量を求める
        # ------------------------------

        # 各スピーカーの遅延量 (d_1, d_2) を推定し, リストに保存
        delays = []
        for csp1_signal, csp2_signal in zip(csp1_values, embedded_freq_weighted_csp_values):

            # CSP1における最初のピーク位置を取得
            # - 第1スピーカーの遅延を示すピークを推定
            csp1_peaks, _ = find_peaks(csp1_signal, height=0.7)
            first_csp1_peak_position = csp1_peaks[0]    # 最初に検出されたピーク位置

            # CSP2の最大値の位置を取得
            # - 第2スピーカーの遅延を示すピークを推定
            csp2_peak_position = np.argmax(csp2_signal)

            # 遅延量 (d_1, d_2) を配列として格納
            # - 推定された各スピーカーの遅延量を保存
            delay_pair = np.array([first_csp1_peak_position, csp2_peak_position])
            delays.append(delay_pair)

        # numpy配列に変更
        # - リスト形式からnumpy配列に変換し, 後続の処理で扱いやすくする
        delays = np.array(delays)

        # 遅延推定誤差を計算する (平均絶対誤差)
        delay_errors = []
        for estimated_delay_pair in delays:
            # 推定された遅延量と基準値 (delay_adjusted_peak_positions) の差を計算
            # - 推定結果が基準値からどれだけずれているかを計算
            error_direct = np.sum(np.abs(estimated_delay_pair - delay_adjusted_peak_positions))

            # 遅延推定ペアの順序が逆の場合の誤差を計算
            # - 順序が異なる場合の誤差も考慮
            error_flipped = np.sum(np.abs(np.flip(estimated_delay_pair) - delay_adjusted_peak_positions))

            # 最小の誤差を選択してリストに追加
            # - 順序が正しい場合と逆の場合で小さい方を採用
            delay_errors.append(np.min([error_direct, error_flipped]))

        # 遅延誤差の平均値を計算
        # - 全ての試行における誤差の平均値を算出
        delay_error_mean = np.mean(np.array(delay_errors))

        # サンプル単位の誤差を時間単位 (ミリ秒) に変換
        # - サンプル単位の誤差を時間単位にして解釈しやすくする
        mean_delay_error_ms = 1000 * (delay_error_mean / sampling_rate)

        # 結果をリストに保存
        # - 遅延推定誤差を蓄積
        delay_time_errors.append(mean_delay_error_ms)

        # ------------------------
        # 22. ピーク比を計算する
        # ------------------------

        # ピーク比 (Peak Ratio) を計算し, 各試行ごとにリストに保存
        peak_ratios = []

        for csp2_signal, estimated_delay_pair in zip(embedded_freq_weighted_csp_values, delays):
            # CSP1が第1スピーカーと第2スピーカーどちらの遅延を検知したか判定
            # - 推定された遅延量と基準値を比較し, 正しい遅延位置を特定
            if np.abs(estimated_delay_pair[0] - delay_adjusted_peak_positions[0]) < np.abs(estimated_delay_pair[0] - delay_adjusted_peak_positions[1]):
                true_delay_position = delay_adjusted_peak_positions[1]  # スピーカーS2からの遅延と推定
            else:
                true_delay_position = delay_adjusted_peak_positions[0]  # スピーカーS1からの遅延と推定

            # 真の遅延位置におけるピーク振幅を取得
            # - 遅延位置に対応するCSP2信号の振幅を取得
            true_peak_amplitude = csp2_signal[true_delay_position]

            # 真の遅延位置以外での最大ピーク振幅を取得
            # - 真の遅延位置をゼロにして次に大きいピークを取得
            csp2_signal_copy = np.copy(csp2_signal)
            csp2_signal_copy[true_delay_position] = 0
            secondary_peak_amplitude = np.max(csp2_signal_copy)

            # ピーク比を計算し, リストに保存
            # - 主ピークの振幅を2番目に大きいピークの振幅で割る
            peak_ratios.append(true_peak_amplitude / (np.max([secondary_peak_amplitude, 1e-8])))

        # リストをnumpy配列に変換
        # - 計算結果を配列形式で保存
        peak_ratios = np.array(peak_ratios)

        # ------------------------------
        # 23. 音質評価 (PESQとSNR)
        # ------------------------------

        # ISTFTを用いて時間波形に変換a
        # - スペクトログラムから時間波形領域を再構築
        num_frames  = min([y1spec.shape[1], y1zero.shape[1]])
        original_waveform = istft(y1spec[:,:num_frames], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        embedded_waveform = istft(y1zero[:,:num_frames], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # PESQ (音質スコア) の計算
        # - 元波形と埋め込み波形を16kHzにダウンサンプリングして評価
        original_waveform_downsampled = resample(original_waveform[:sampling_rate * 5], orig_sr=sampling_rate, target_sr=16000)
        embedded_waveform_downsampled = resample(embedded_waveform[:sampling_rate * 5], orig_sr=sampling_rate, target_sr=16000)
        pesq_score = pesq(16000, original_waveform_downsampled, embedded_waveform_downsampled)
        pesq_scores.append(pesq_score)

        # SNR (信号対雑音比) の計算
        # - 元波形と埋め込み波形の差を基にSNRを算出
        signal_power = sum(original_waveform ** 2)
        noise_power  = sum((original_waveform - embedded_waveform) ** 2)
        snr = 20 * np.log10(signal_power / noise_power)

        # 生成された音源の保存
        # - 埋め込み強度を含むファイル名で波形を保存
        sf.write(f'./../../sound/distance_estimation/music{music_type}_mono/embedded_music{music_type}_gain{amplitude_gain:.2f}.wav', embedded_waveform, sampling_rate)

        # 確認用の進捗表示
        print(f'{(int(num+1) / loop_times)*100:3.0f}% Completed')

    # numpy配列に変換
    # - 遅延推定誤差と音質スコアを配列形式で保存
    delay_time_errors = np.array(delay_time_errors)
    pesq_scores = np.array(pesq_scores)


    # ------------------------------
    # 24. CSV形式で計算結果を出力
    # ------------------------------

    # 保存先のパスを指定
    # - ログファイル
    logs_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/logs'

    # - 計算結果
    raw_data_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/raw_data'

    # ログファイルに書き込むデータ
    # - 1_1. 設定条件
    setting_parameters_data = [
        ["設定条件"],     # パラメータ設定の見出し
        ["L: 埋め込み周波数のビン数 [bin]", "合計の周波数ビン数 [bin]", "M: 埋め込みを行うフレーム数", "フレームをずらす回数"],
        [f"{embedding_frequency_bins}", f"{frame_length + 1}", f"{num_embedding_frames}", f"{len(pos_st_frame)}"]
    ]

    # - 1_2. 書き込み処理
    with open(f'{logs_path}/settings_parameters.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(setting_parameters_data)
    
    # - 2_1. スピーカー・マイク距離, 到来時間
    distance_and_arrival_data = [
        ["スピーカー・マイク距離, 到来時間"],     # パラメータ設定の見出し
        ["スピーカー1_距離[m]", "スピーカー1_到来時間[ms]", "スピーカー2_距離[m]", "スピーカー2_到来時間[ms]"],
        [f"{first_detected_peak_positions[0]/sampling_rate*sound_of_speed:.2f}", f"{1000*first_detected_peak_positions[0]/sampling_rate:.2f}", f"{first_detected_peak_positions[1]/sampling_rate*sound_of_speed:.2f}", f"{1000*first_detected_peak_positions[1]/sampling_rate:.2f}"]
    ]
    # - 2_2. 書き込み処理
    with open(f'{logs_path}/distance_and_arrival.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(distance_and_arrival_data)

    # リストをCSV形式で書き出し
    # - 計算されたデータをそれぞれのファイルに保存
    save.to_csv(x_0, f'music{music_type}_mono_trimmed', f'{raw_data_path}/music{music_type}_mono_trimmed.csv')
    save.to_csv(impulse, 'impulse', f'{raw_data_path}/impulse.csv')
    save.to_csv(first_detected_peak_positions, 'first_detected_peak_positions', f'{raw_data_path}/first_detected_peak_positions.csv')
    save.to_csv(delay_adjusted_peak_positions, 'delay_adjusted_peak_positions', f'{raw_data_path}/delay_adjusted_peak_positions.csv')
    save.to_csv(csp1_values, 'csp1_values', f'{raw_data_path}/csp1_values.csv')
    save.to_csv(csp2_values, 'csp2_values', f'{raw_data_path}/csp2_values.csv')
    save.to_csv(store_embedded_frequencies, 'embedded_frequencies', f'{raw_data_path}/embedded_frequencies.csv')
    save.to_csv(embedded_freq_csp1_values, 'embedded_freq_csp1_values', f'{raw_data_path}/embedded_freq_csp1_values.csv')
    save.to_csv(embedded_freq_csp2_values, 'embedded_freq_csp2_values', f'{raw_data_path}/embedded_freq_csp2_values.csv')
    save.to_csv(csp_difference_values, 'csp_difference_values', f'{raw_data_path}/csp_difference_values.csv')
    save.to_csv(weighted_csp_difference_values, 'weighted_csp_difference_values', f'{raw_data_path}/weighted_csp_difference_values.csv')
    save.to_csv(embedded_freq_csp_difference, 'embedded_freq_csp_difference', f'{raw_data_path}/embedded_freq_csp_difference.csv')
    save.to_csv(embedded_freq_weighted_csp_values, 'embedded_freq_weighted_csp_values', f'{raw_data_path}/embedded_freq_weighted_csp_values.csv')
    save.to_csv(delay_time_errors, 'delay_time_errors', f'{raw_data_path}/delay_time_errors.csv')
    save.to_csv(pesq_scores, 'pesq_scores', f'{raw_data_path}/pesq_scores.csv')
