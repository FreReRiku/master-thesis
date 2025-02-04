""" 
visualize.py
------------

CSVファイルを用いてグラフを作成します.

Created by FreReRiku on 2025/01/17
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_embedding_error(music_type, emb_type, variable):
    """
    埋込強度変化に伴う推定誤差とPESQの変化をプロットする関数.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    emb_type: str
        埋め込みタイプの識別子. "amplitude_modulation" または "phase_modulation" を想定.
    
    Returns
    -------
    None
    """
    
    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 変数設定
    n_fft = 2048
    fs = 44100
    time_axis = np.arange(n_fft) / fs  # ※今回のプロットには使用していないが, 他で必要なら保持
    embedding_amplitudes = np.linspace(0, 1, 25)
    speed_of_sound = 340.29

    # emb_type によって x 軸の値とラベル, 表示範囲を設定
    if emb_type == "amplitude_modulation":
        x_values = embedding_amplitudes
        x_label = "Embedding Amplitude Gain"
        x_limits = [-0.05, 1.0]
        legend_loc = 'upper left'
    elif emb_type == "phase_modulation": 
        # 0～1 の振幅を 0～180° に線形変換
        x_values = embedding_amplitudes * 180
        x_label = "Embedding Phase [degree]"
        x_limits = [0, 180]
        legend_loc = 'upper right'
    # 使用するCSVファイルのパスを指定
    delay_time_errors_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/delay_time_errors.csv'
    pesq_scores_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/pesq_scores.csv'
    
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
    ax1.plot(x_values, delay_time_errors * speed_of_sound / 1000, label='Distance Estimation Error')
    ax1.set_xlim(x_limits)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Estimation Distance Error [m]")
    
    # 右縦軸プロット (PESQスコア)
    ax2 = ax1.twinx()
    ax2.plot(x_values, pesq_scores, 'r', label='PESQ')
    ax2.set_ylim([-0.05, 5.5])
    ax2.set_ylabel("PESQ")
    
    # 凡例の設定
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
    
    # 画像の保存
    filename = f'{output_path}/{emb_type}_vs_PESQ.svg'
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    
    plt.close()
    # 必要なら plt.show() を有効化
    # plt.show()
    
    return


def plot_impulse(music_type, emb_type, variable):
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

    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 変数設定
    fft_points = 2048
    fs = 44100
    time_axis = np.arange(fft_points) / fs

    # 使用するCSVファイルのパスを指定
    impulse_position_data_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/first_detected_peak_positions.csv'
    impulse_response_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/impulse.csv'

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
    filename = f'{output_path}/impulse.svg'
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.close()

    return
    

def plot_mean_embedded_csp(music_type, emb_type, variable):
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

    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2
    
    # 使用するCSVファイルのパスを指定
    raw_data_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data'
    # embedded_csp1_path = f'{raw_data_path}/embedded_freq_csp1_values.csv'
    csp1_path = f'{raw_data_path}/csp1_values.csv'
    embedded_csp2_path = f'{raw_data_path}/csp2_values.csv'
    # embedded_csp2_path = f'{raw_data_path}/embedded_freq_csp2_values.csv'
    embedded_subtract_csp_path = f'{raw_data_path}/embedded_freq_csp_difference.csv'
    embedded_weighted_csp_path = f'{raw_data_path}/embedded_freq_weighted_csp_values.csv'
    delay_adjusted_peak_positions_path = f'{raw_data_path}/delay_adjusted_peak_positions.csv'
    
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        # 2次元データの読み込み
        csp1 = pd.read_csv(csp1_path, header=None, skiprows=1).to_numpy()
        # embedded_csp1 = pd.read_csv(embedded_csp1_path, header=None, skiprows=1).to_numpy()
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
    mean_csp1          = np.mean(csp1[:, :fft_points], axis=0)
    # mean_emb_freq_csp1          = np.mean(embedded_csp1[:, :fft_points], axis=0)
    mean_emb_freq_csp2          = np.mean(embedded_csp2[:, :fft_points], axis=0)
    mean_emb_freq_subtract_csp  = np.mean(embedded_subtract_csp[:, :fft_points], axis=0)
    mean_emb_freq_weighted_csp  = np.mean(embedded_weighted_csp[:, :fft_points], axis=0)
    
    for p, c in zip(delay_adjusted_peak_positions, ['r', 'g']):
        for ax in [ax1, ax2, ax3]:
            ax.axvline(1000*p/fs, color=c, linestyle='--')
        for ax in [ax1, ax2]:
            ax.axhline(threshold_ratio, color='k', linestyle=':')
    ax1.plot(1000*time_axis, mean_csp1)
    # ax1.plot(1000*time_axis, mean_emb_freq_csp1)
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
    filename = f'{output_path}/csp.svg'
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.close()
    
    return

def plot_embedded_frequencies(music_type, emb_type, variable):

    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 変数設定
    frame_length    = 1024
    sampling_rate   = 44100

    # 埋め込み周波数のCSVファイルのパス
    data_file_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/embedded_frequencies.csv'

    # CSVファイルを読み込み
    data_file = pd.read_csv(data_file_path)

    # ヘッダー行を取り除き, 埋め込まれた周波数を取得
    embedded_frequencies = data_file["embedded_frequencies"]

    # 範囲外のインデックスを除外
    valid_indices = embedded_frequencies[embedded_frequencies < (frame_length // 2 + 1)]

    # 周波数ビンから実際の周波数スケールに変換
    freq_bins = np.linspace(0, sampling_rate / 2, frame_length // 2 + 1)
    frequencies = freq_bins[valid_indices]

    # 描画
    plt.figure(figsize=(12, 6))
    plt.hist(frequencies, bins=50, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Embedded Frequencies")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Count")
    plt.grid(True)

    
    # 保存
    filename = f'{output_path}/embedded_frequencies.svg'
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.close()

    return

def plot_audio_waveform(music_type, emb_type, variable):

    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 変数設定
    sampling_rate   = 44100

    # 埋め込み周波数のCSVファイルのパス
    data_file_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data/music{music_type}_mono_trimmed.csv'

    # CSVファイルを読み込み
    data_file = pd.read_csv(data_file_path)

    # ヘッダー行を取り除き, 埋め込まれた周波数を取得
    amplitude = data_file.iloc[:, 0]

    # 時間軸の生成
    duration = len(amplitude) / sampling_rate
    time = np.linspace(0, duration, len(amplitude))


    # 描画
    plt.figure(figsize=(12, 6))
    plt.plot(time, amplitude, color='blue')
    plt.title("Sound_signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    
    # 保存
    filename = f'{output_path}/audio_waveform.svg'
    plt.savefig(filename)
    print(f"画像が保存されました: {filename}")
    plt.close()

    return

def plot_mean_csp1(music_type, emb_type, variable):
    """
    CSP1グラフを作成する関数.
    
    この関数は, 与えられた音楽データの埋め込みCSPに基づく平均的なグラフをプロットし, 結果を保存します.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    emb_type: str
        埋め込みタイプの識別子.
    
    Returns
    -------
    None
    """
    
    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2
    
    # 使用するCSVファイルのパスを指定
    raw_data_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data'
    csp1_path = f'{raw_data_path}/csp1_values.csv'
    delay_adjusted_peak_positions_path = f'{raw_data_path}/delay_adjusted_peak_positions.csv'
    
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        csp1 = pd.read_csv(csp1_path, header=None, skiprows=1).to_numpy()
        delay_adjusted_peak_positions = pd.read_csv(delay_adjusted_peak_positions_path, header=None, skiprows=1).squeeze("columns").to_numpy()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # 図全体に1つのサブプロットを配置するために, グリッド指定 (1, 1, 1) とし, figsize も横長に設定する
    fig = plt.figure(num='CSP1', figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    
    # 平均CSP1を計算
    mean_csp1 = np.mean(csp1[:, :fft_points], axis=0)
    
    # 遅延調整されたピーク位置に対して, 各ピーク位置で縦線を描画する
    for p, c in zip(delay_adjusted_peak_positions, ['r', 'g']):
        ax1.axvline(1000 * p / fs, color=c, linestyle='--')
    
    # 閾値ラインは1回だけ描画する
    ax1.axhline(threshold_ratio, color='k', linestyle=':')
    
    # 平均CSP1のプロット
    ax1.plot(1000 * time_axis, mean_csp1, color='b')
    
    # 軸ラベルとタイトルの設定
    ax1.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("CSP1")
    ax1.set_xlim([1000 * time_axis[0], 1000 * time_axis[-1]])
    ax1.set_ylim([-0.5, 1.1])
    
    # レイアウトを自動調整して余白を最適化する
    plt.tight_layout()
    
    # 画像の保存
    filename = f'{output_path}/csp1.svg'
    plt.savefig(filename, bbox_inches='tight')
    print(f"画像が保存されました: {filename}")
    plt.close()
    
    return

def plot_mean_csp2(music_type, emb_type, variable):
    """
    CSP1グラフを作成する関数.
    
    この関数は, 与えられた音楽データの埋め込みCSPに基づく平均的なグラフをプロットし, 結果を保存します.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    emb_type: str
        埋め込みタイプの識別子.
    
    Returns
    -------
    None
    """
    
    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2
    
    # 使用するCSVファイルのパスを指定
    raw_data_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data'
    csp2_path = f'{raw_data_path}/csp2_values.csv'
    delay_adjusted_peak_positions_path = f'{raw_data_path}/delay_adjusted_peak_positions.csv'
    
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        csp2 = pd.read_csv(csp2_path, header=None, skiprows=1).to_numpy()
        delay_adjusted_peak_positions = pd.read_csv(delay_adjusted_peak_positions_path, header=None, skiprows=1).squeeze("columns").to_numpy()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # 図全体に1つのサブプロットを配置するために, グリッド指定 (1, 1, 1) とし, figsize も横長に設定する
    fig = plt.figure(num='CSP1', figsize=(8, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    
    # 平均CSP1を計算
    mean_csp2 = np.mean(csp2[:, :fft_points], axis=0)
    
    # 遅延調整されたピーク位置に対して, 各ピーク位置で縦線を描画する
    for p, c in zip(delay_adjusted_peak_positions, ['r', 'g']):
        ax1.axvline(1000 * p / fs, color=c, linestyle='--')
    
    # 閾値ラインは1回だけ描画する
    ax1.axhline(threshold_ratio, color='k', linestyle=':')
    
    # 平均CSP1のプロット
    ax1.plot(1000 * time_axis, mean_csp2, color='b')
    
    # 軸ラベルとタイトルの設定
    ax1.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("CSP2")
    ax1.set_xlim([1000 * time_axis[0], 1000 * time_axis[-1]])
    ax1.set_ylim([-0.5, 1.1])
    
    # レイアウトを自動調整して余白を最適化する
    plt.tight_layout()
    
    # 画像の保存
    filename = f'{output_path}/csp2.svg'
    plt.savefig(filename, bbox_inches='tight')
    print(f"画像が保存されました: {filename}")
    plt.close()
    
    return

def plot_mean_csp_ws(music_type, emb_type, variable):
    """
    埋め込み周波数を利用したCSPグラフを作成する関数.
    
    この関数は, 与えられた音楽データの埋め込みCSPに基づく平均的なグラフをプロットし, 結果を保存します.
    
    Parameters
    ----------
    music_type: int
        使用する楽曲のタイプ.
    emb_type: str
        埋め込みタイプの識別子.
    
    Returns
    -------
    None
    """
    
    # 必要なディレクトリを作成
    output_path = f'./../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 変数設定
    fft_points = 1024
    fs = 44100
    time_axis = np.arange(fft_points) / fs
    threshold_ratio = 0.2
    
    # 使用するCSVファイルのパスを指定
    raw_data_path = f'./../../data/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/csv_files/raw_data'
    embedded_subtract_csp_path = f'{raw_data_path}/embedded_freq_csp_difference.csv'
    embedded_weighted_csp_path = f'{raw_data_path}/embedded_freq_weighted_csp_values.csv'
    delay_adjusted_peak_positions_path = f'{raw_data_path}/delay_adjusted_peak_positions.csv'
    
    # --------------------
    # CSVファイルの読み込み
    # --------------------
    try:
        embedded_subtract_csp = pd.read_csv(embedded_subtract_csp_path, header=None, skiprows=1).to_numpy()
        embedded_weighted_csp = pd.read_csv(embedded_weighted_csp_path, header=None, skiprows=1).to_numpy()
        delay_adjusted_peak_positions = pd.read_csv(delay_adjusted_peak_positions_path, header=None, skiprows=1).squeeze("columns").to_numpy()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # plt.subplots を用いて, figsize=(8,4) の横長レイアウトで図と軸を生成
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    
    # 平均値の計算
    mean_emb_freq_subtract_csp = np.mean(embedded_subtract_csp[:, :fft_points], axis=0)
    mean_emb_freq_weighted_csp = np.mean(embedded_weighted_csp[:, :fft_points], axis=0)
    
    # 遅延調整されたピーク位置に対応する縦線の描画
    for p, c in zip(delay_adjusted_peak_positions, ['r', 'g']):
        ax1.axvline(1000 * p / fs, color=c, linestyle='--')
    
    # 各計算結果のプロット
    ax1.plot(1000 * time_axis, mean_emb_freq_subtract_csp, color='lightgray')
    ax1.plot(1000 * time_axis, mean_emb_freq_weighted_csp, color='r')
    
    # 軸ラベル, タイトル, および表示範囲の設定
    ax1.set_xlabel("Time [ms]", fontname="MS Gothic")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Weighted Difference-CSP")
    ax1.set_xlim([1000 * time_axis[0], 1000 * time_axis[-1]])
    _, y_max = ax1.get_ylim()
    ax1.set_ylim([0, y_max])
    
    # 自動レイアウト調整
    plt.tight_layout()
    
    # 画像の保存
    filename = f'{output_path}/csp_ws.svg'
    plt.savefig(filename, bbox_inches='tight')
    print(f"画像が保存されました: {filename}")
    plt.close()
    
    return

def plot_AM_vs_PM(music_type, emb_type, variable):
    """
    Returns
    -------
    None
    """
    
    # 出力パスの設定, (該当ディレクトリがなければ自動で作成)
    output_path = f'../../figure/distance_estimation/music{music_type}_mono/var_{variable}/{emb_type}/'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 日本語対応フォントの設定（システムに合わせて適宜変更してください）
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['IPAexGothic', 'Yu Gothic', 'TakaoPGothic', 'Meirio', 'MS Gothic']

    # 基本パラメータの設定
    n_fft = 2048
    fs = 44100
    time_axis = np.arange(n_fft) / fs  # 今回は使用しないが保持
    embedding_amplitudes = np.linspace(0, 1, 25)  # 元の埋込パラメータ t
    amplitude_gain = 1 - embedding_amplitudes       # 下部x軸は1から0へ表示
    speed_of_sound = 340.29
    
    # CSVファイルのパス設定
    am_delay_path = f'../../data/distance_estimation/music{music_type}_mono/var_{variable}/amplitude_modulation/csv_files/raw_data/delay_time_errors.csv'
    pm_delay_path = f'../../data/distance_estimation/music{music_type}_mono/var_{variable}/phase_modulation/csv_files/raw_data/delay_time_errors.csv'
    am_pesq_path = f'../../data/distance_estimation/music{music_type}_mono/var_{variable}/amplitude_modulation/csv_files/raw_data/pesq_scores.csv'
    pm_pesq_path = f'../../data/distance_estimation/music{music_type}_mono/var_{variable}/phase_modulation/csv_files/raw_data/pesq_scores.csv'
    
    # CSVファイルの読み込み
    try:
        # AM_delay_time_errors の値の順序を逆転
        am_delay = pd.read_csv(am_delay_path, header=None).squeeze("columns")
        am_delay = pd.to_numeric(am_delay, errors='coerce').dropna().to_numpy()
        am_delay = am_delay[::-1]
        
        pm_delay = pd.read_csv(pm_delay_path, header=None).squeeze("columns")
        pm_delay = pd.to_numeric(pm_delay, errors='coerce').dropna().to_numpy()
        
        # AM_pesq_scores の値の順序を逆転
        am_pesq = pd.read_csv(am_pesq_path, header=None).squeeze("columns")
        am_pesq = pd.to_numeric(am_pesq, errors='coerce').dropna().to_numpy()
        am_pesq = am_pesq[::-1]
        
        pm_pesq = pd.read_csv(pm_pesq_path, header=None).squeeze("columns")
        pm_pesq = pd.to_numeric(pm_pesq, errors='coerce').dropna().to_numpy()
        
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # 図の作成 (figsize を大きめに設定)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35)
    
    # 推定誤差の計算 (距離[m]に変換)
    am_distance_error = am_delay * speed_of_sound / 1000
    pm_distance_error = pm_delay * speed_of_sound / 1000
    
    # プロット（Lineオブジェクトを変数に格納）
    # ax1：推定誤差（実線）
    line_am_error, = ax1.plot(amplitude_gain, am_distance_error, color='blue', linestyle='-', 
                              label='AM推定誤差: 青実線')
    line_pm_error, = ax1.plot(amplitude_gain, pm_distance_error, color='red', linestyle='-', 
                              label='PM推定誤差: 赤実線')
    ax1.set_xlim(1.0, 0.0)
    ax1.set_xlabel("Embedding Amplitude Gain", fontsize=20)
    ax1.set_ylabel("Estimation Distance Error [m]", fontsize=20)
    
    # ax2：PESQスコア（破線）
    ax2 = ax1.twinx()
    line_am_pesq, = ax2.plot(amplitude_gain, am_pesq, color='blue', linestyle='--', 
                             label='AM音質: 青破線')
    line_pm_pesq, = ax2.plot(amplitude_gain, pm_pesq, color='red', linestyle='--', 
                             label='PM: 赤破線')
    ax2.set_ylim(-0.05, 5.5)
    ax2.set_ylabel("PESQ", fontsize=20)
    
    # 副x軸の追加: (1 - A)*180 で「Embedding Phase [degree]」を表示
    secax = ax1.secondary_xaxis('top',
                                functions=(lambda A: (1 - A) * 180,
                                           lambda P: 1 - P / 180))
    secax.set_xlabel("Embedding Phase [degree]", fontsize=20)
    
    # 凡例を手動で順序通りに設定:
    custom_handles = [line_am_error, line_am_pesq, line_pm_error, line_pm_pesq]
    custom_labels  = ["AM推定誤差: 青実線", "AM音質: 青破線", "PM推定誤差: 赤実線", "PM: 赤破線"]
    fig.legend(custom_handles, custom_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.2), ncol=4, fontsize=16)
    
    # 画像の保存 (bbox_inches='tight' を指定して余白も含める)
    filename = f'{output_path}/combined_vs_PESQ.svg'
    plt.savefig(filename, bbox_inches='tight')
    print(f"画像が保存されました: {filename}")
    
    plt.close()
    return
