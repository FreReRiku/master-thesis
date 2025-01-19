""" 
visualize.py
------------

CSVファイルを用いてグラフを作成します.

Created by FreReRiku on 2025/01/17
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_embedding_error(music_type):

    """
    埋込強度変化に伴う推定誤差とPESQの変化をプロットする関数.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    
    Returns
    -------
    None
    
    """
    
    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2

    embedding_amplitudes = np.linspace(0, 1, 25)
    speed_of_sound=340.29

    # 使用するCSVファイルのパスを指定
    delay_time_errors_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/delay_time_errors.csv'
    pesq_scores_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/pesq_scores.csv'
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        delay_time_errors = pd.read_csv(delay_time_errors_path, header=None).squeeze("columns")
        delay_time_errors = pd.to_numeric(delay_time_errors, errors='coerce').dropna().to_numpy()

        pesq_scores = pd.read_csv(pesq_scores_path, header=None).squeeze("columns")
        pesq_scores = pd.to_numeric(pesq_scores, errors='coerce').dropna().to_numpy()

    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # 図を作成
    fig = plt.figure(num='埋め込み強度変化', figsize=(6, 3))
    plt.subplots_adjust(bottom=0.15)
    ax1 = fig.add_subplot(1, 1, 1)
    
    # 左縦軸プロット (推定距離誤差)
    ax1.plot(embedding_amplitudes, delay_time_errors * speed_of_sound / 1000, label='Distance Estimation Error')
    ax1.set_xlim([-0.05, 1.0])
    ax1.set_xlabel("Embedding Amplitude Gain")
    ax1.set_ylabel("Estimation Distance Error [m]")
    
    # 右縦軸プロット (PESQスコア)
    ax2 = ax1.twinx()
    ax2.plot(embedding_amplitudes, pesq_scores, 'r', label='PESQ')
    ax2.set_ylim([-0.05, 5.5])
    ax2.set_ylabel("PESQ")
    
    # 凡例の設定
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # 保存
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/plot_embedding_error'
    filename = f"{output_path}/Amp_vs_PESQ.svg"
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    
    plt.clf()
    # 表示 (必要ならコメントアウト解除で有効化してください)
    # plt.show()
    
    return

def plot_impulse(music_type):
    """
    インパルス応答をプロットする関数.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    
    Returns
    -------
    None
    
    """

    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2

    # 使用するCSVファイルのパスを指定
    impulse_position_data_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/first_detected_peak_positions.csv'
    impulse_response_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/impulse.csv'

    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        impulse_position_data = pd.read_csv(impulse_position_data_path, header=None).squeeze("columns")
        impulse_position_data = pd.to_numeric(impulse_position_data, errors='coerce').dropna().to_numpy()

        impulse_response = pd.read_csv(impulse_response_path, header=None).squeeze("columns")
        impulse_response = pd.to_numeric(impulse_response, errors='coerce').dropna().to_numpy()

    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # --------------------
    # インパルス応答のプロット
    # --------------------
    
    fig = plt.figure(num='インパルス応答', figsize=(6, 3))
    plt.subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(1, 1, 1)
    for p, c in zip(impulse_position_data, ['r', 'g']):
        ax.axvline(1000 * p / fs, color=c, linestyle='--')
    ax.plot(1000*time_axis, impulse_response[:fft_points])
    ax.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax.set_ylabel("Amplitude")
    ax.set_title('Impulse')
    ax.set_xlim([1000*time_axis[0], 1000*time_axis[-1]])

    # 保存
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/plot_impulse'
    filename = f"{output_path}/impulse.svg"
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.clf()

    return
    

def plot_mean_embedded_csp(music_type):
    """
    埋め込み周波数を利用したCSPグラフを作成する関数.
    
    この関数は、与えられた音楽データの埋め込みCSPに基づく平均的なグラフをプロットし、結果を保存します。
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    
    Returns
    -------
    None
    
    """
    
    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2
    
    # 使用するCSVファイルのパスを指定
    embedded_csp1_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/embedded_freq_csp1_values.csv'
    embedded_csp2_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/embedded_freq_csp2_values.csv'
    embedded_subtract_csp_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/embedded_freq_csp_difference.csv'
    embedded_weighted_csp_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/embedded_freq_weighted_csp_values.csv'
    delay_adjusted_peak_positions_path = f'./../../data/distance_estimation/music{music_type}_mono/csv_files/delay_adjusted_peak_positions.csv'
    
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        # 2次元データの読み込み
        embedded_csp1 = pd.read_csv(embedded_csp1_path, header=None, skiprows=1).to_numpy()
        embedded_csp2 = pd.read_csv(embedded_csp2_path, header=None, skiprows=1).to_numpy()
        embedded_subtract_csp = pd.read_csv(embedded_subtract_csp_path, header=None, skiprows=1).to_numpy()
        embedded_weighted_csp = pd.read_csv(embedded_weighted_csp_path, header=None, skiprows=1).to_numpy()
        
        # 1列データの読み込み
        delay_adjusted_peak_positions = pd.read_csv(delay_adjusted_peak_positions_path, header=None, skiprows=1).squeeze("columns").to_numpy()
    
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    fig = plt.figure(num='CSP(埋め込み周波数のみ)', figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    
    # 各計算結果をプロット
    mean_emb_freq_csp1          = np.mean(embedded_csp1[:, :fft_points], axis=0)
    mean_emb_freq_csp2          = np.mean(embedded_csp2[:, :fft_points], axis=0)
    mean_emb_freq_subtract_csp  = np.mean(embedded_subtract_csp[:, :fft_points], axis=0)
    mean_emb_freq_weighted_csp  = np.mean(embedded_weighted_csp[:, :fft_points], axis=0)
    
    for p, c in zip(delay_adjusted_peak_positions, ['r', 'g']):
        for ax in [ax1, ax2, ax3]:
            ax.axvline(1000*p/fs, color=c, linestyle='--')
        for ax in [ax1, ax2]:
            ax.axhline(threshold_ratio, color='k', linestyle=':')
    ax1.plot(1000*time_axis, mean_emb_freq_csp1)
    ax2.plot(1000*time_axis, mean_emb_freq_csp2)
    ax3.plot(1000*time_axis, mean_emb_freq_subtract_csp, 'lightgray')
    ax3.plot(1000*time_axis, mean_emb_freq_weighted_csp, 'r')
    
    # 各subplotにラベルを追加
    ax1.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("CSP1 (without embedding)")
    ax1.set_xlim([1000*time_axis[0], 1000*time_axis[-1]])
    ax1.set_ylim([-0.5, 1.1])
    
    ax2.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("CSP2(with embedding)")
    ax2.set_xlim([1000*time_axis[0], 1000*time_axis[-1]])
    ax2.set_ylim([-0.5, 1.1])
    
    ax3.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Weighted Difference-CSP")
    ax3.set_xlim([1000*time_axis[0], 1000*time_axis[-1]])
    _, y_max = ax3.get_ylim()
    ax3.set_ylim([0, y_max])
    
    # 保存
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/plot_csp'
    filename = f"{output_path}/csp.svg"
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.clf()
    
    return
