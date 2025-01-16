""" 
visualize.py
------------

CSVファイルを用いてグラフを作成します.

Created by FreReRiku on 2025/01/17
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_embedding_error(
        delay_time_errors_file,
        pesq_scores_file,
        embedding_amplitudes,
        speed_of_sound,
        music_type,
        num='埋め込み強度変化',
        figsize=(6, 3),
        output_path='./../../figure/distance_estimation/plot_embedding_error/'):

    """
    埋込強度変化に伴う推定誤差とPESQの変化をプロットする関数.
    
    Parameters
    ----------
    embedding_amplitudes: list or array
        埋込強度ゲインのリスト.
    delay_time_errors: str
        遅延時間誤差を含むCSVファイルのパス.
    pesq_scores: list or array
        PESQスコアを含むCSVファイルのパス.
    speed_of_sound: float
        音速.
    music_type=music_type: int
        使用する楽曲のタイプ. (デフォルト: main.py内のmusic_typeと同じ値を使用します.)
    num: str
        プロットのウィンドウ名.
    figsize: tuple
        図のサイズ. (デフォルト: (6, 3))
    output_path: str
        保存先ディレクトリのパス.
    
    Returns
    -------
    None
    
    """
    
    # CSVファイルを読み込む (ヘッダー行をスキップ)
    delay_data = pd.read_csv(delay_time_errors_file, header=0)
    pesq_data = pd.read_csv(pesq_scores_file, header=0)

    # 必要なデータを取得 (列インデックスを指定)
    delay_time_errors = delay_data.iloc[:, 0]   # 1列目: delay_time_errors
    pesq_scores = pesq_data.iloc[:, 0]          # 1列目: pesq_scores

    # 図を作成
    fig = plt.figure(num=num, figsize=figsize)
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
    filename = f"{output_path}/music{music_type}_Amp_vs_PESQ.png"
    plt.savefig(filename)
    print(f"Figure saved to {filename}")
    
    # 表示 (必要ならコメントアウト解除で有効化してください)
    # plt.show()
    
    return
