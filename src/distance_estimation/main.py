'''----------
distance_estimation.py
埋込強度の変化に伴う推定誤差結果の変化をプロット
Created by FreReRiku on 2024/12/30
----------'''

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pesq import pesq
from librosa import stft, istft, resample
from scipy.signal import find_peaks
from scipy.fft import irfft

# 音源の選択 (1 or 2)
music_type = 1

# --パラメータ----------
# サンプリング周波数 [Hz]
fs  = 44100
# 音速 [m/s]
c   = 340.29
# サンプル長 (fs * s)
L   = fs*10
# フレーム長
N   = 1024
# ホップ長
S   = 512


# スタートポイント
st = 1000
# エンドポイント
ed = st + L

# トライアル回数
Tn = 100

# 連続して埋め込むフレーム数
K = 40
# 埋め込み周波数のビン数
D   = np.floor(N*0.1).astype(int)
# スタートのフレーム位置(ここからKフレーム用いる)
pos_st_frame = np.arange(0, Tn*3, 3)
# CSPの最大値に対するノイズと判定する振幅比のしきい値
TH  = 0.2

# 埋め込む振幅の設定
# ループ回数
loop_times = 25
emb_amp    = np.linspace(0, 1, loop_times)
# 埋め込む位相の設定
emb_phase  = 0

# ゼロ除算回避定数
eps = 1e-20

dte_data, pesq_data = [], []

print('[ 設定条件 ]')
print(f' - ゼロを埋め込む周波数ビンの数：{D}bin/{N+1}bin中')
print(f' - １回の検知で埋め込むフレーム数：{K}フレーム')
print(f' - 試行回数：{len(pos_st_frame)}回\n')


for num, amp in enumerate(emb_amp):

    # --オーディオファイルの読み込み----------
    # ファイル名
    file_name_impulse1  = f'./../../sound_data/room_simulation/impulse_signal_ch1_{fs}Hz.wav'
    file_name_impulse2  = f'./../../sound_data/room_simulation/impulse_signal_ch2_{fs}Hz.wav'
    file_name_origin    = f'./../../sound_data/original/music{music_type}_mono.wav'
    file_name_spk1 = f'./../../sound_data/room_simulation/music{music_type}_room_ch1_{fs}Hz.wav'
    file_name_spk2 = f'./../../sound_data/room_simulation/music{music_type}_room_ch2_{fs}Hz.wav'
    # 読み込み
    impulse1, _ = sf.read(file_name_impulse1)
    impulse2, _ = sf.read(file_name_impulse2)
    x, _        = sf.read(file_name_origin)
    y1, _       = sf.read(file_name_spk1)
    y2, _       = sf.read(file_name_spk2)

    # 時間軸
    t = np.arange(N)/fs

    # 音声のトリミング (長すぎるので)
    x_0       = x[st:ed]          # スピーカ出力音声のトリミング
    y1_0      = y1[st:ed]         # マイク入力音声1のトリミング
    y2_0      = y2[st:ed]         # マイク入力音声1のトリミング

    # スペクトログラム
    Xspec   = stft(x_0, n_fft=2*N, hop_length=S, win_length=N, center=False)
    Y1spec  = stft(y1_0, n_fft=2*N, hop_length=S, win_length=2*N, center=False)
    # Y1specのサイズを調べたい。
    # print("Y1specのサイズ:", Y1spec.shape)

    # print(f'Y1spec: {Y1spec}')
    Y2spec  = stft(y2_0, n_fft=2*N, hop_length=S, win_length=2*N, center=False)
    Y1zero  = stft(y1, n_fft=2*N, hop_length=S, win_length=2*N, center=False)

    # 保存用の配列
    CSP0_data, CSP_data, CSP1_data, CSP2_data, CSP_emb_data, CSP_sub_data, CSP_wtd_data , CSP_emb_sub_data, CSP_emb_wtd_data = [], [], [], [], [], [], [], [], []


    # 1st: CSP0, 及びTop Position d_0 の推定
    for k in pos_st_frame:

        # マイク入力音声のスペクトログラム
        Yspec = Y1spec + Y2spec

        # --CSP0を求める(GCC-PHAT法)----------
        # 相互相関(周波数領域)
        XY0 = Yspec[:, k:k + K] * np.conj(Xspec[:, k:k + K])
        # 相互相関の絶対値(周波数領域)
        XY0abs = np.abs(XY0)
        # 分母がほぼ0になるのを防止
        XY0abs[XY0abs < eps] = eps
        # 白色化相互相関(周波数領域)
        CSP0_sp = XY0 / XY0abs
        # 時間方向で平均
        CSP0 = np.mean(CSP0_sp, axis=1)
        # 逆STFT
        CSP0_ave = irfft(CSP0, axis=0)

        # -CSP0の整形-----
        # いらない後半部を除去
        CSP0_ave = CSP0_ave[:N]
        # 最大で割り算
        CSP0_ave = CSP0_ave / np.max(CSP0_ave)

        # -CSP0の保存-----
        CSP0_data.append(CSP0_ave)

        # -dを推定-----
        d = (np.argmax(CSP0_data)-25)


    # --インパルスのピーク位置の推定----------
    pos_imp = []
    pos_imp_sub_d = []
    for impulse_ in [impulse1, impulse2]:
        pos_peaks, _ = find_peaks(impulse_, height=0.2)
        pos_imp.append(pos_peaks[0])
        pos_imp_sub_d.append(pos_peaks[0]-d)

    pos_imp = np.array(pos_imp)
    pos_imp_sub_d = np.array(pos_imp_sub_d)

    # 音声のトリミング
    x       = x[st-d:ed-d]

    # スペクトログラム
    Xspec   = stft(x, n_fft=2*N, hop_length=S, win_length=N, center=False)

    for k in pos_st_frame:

        # 埋め込み用の配列
        Y1emb   = np.copy(Y1spec)

        # マイク入力音声のスペクトログラム
        Yspec       = Y1spec + Y2spec

        # --CSP1を求める----------
        XY1          = Yspec[:, k:k+K] * np.conj(Xspec[:, k:k+K])    # 相互相関(周波数領域)
        XY1abs       = np.abs(XY1)                # 相互相関の絶対値(周波数領域)
        XY1abs[XY1abs < eps] = eps                # 分母がほぼ0になるのを防止
        CSP1_sp      = XY1/XY1abs                  # 白色化相互相関(周波数領域)
        CSP1         = np.mean(CSP1_sp, axis=1)   # 時間方向で平均
        CSP1_ave     = irfft(CSP1, axis=0)        # 逆STFT

        # CSPの整形
        CSP1_ave     = CSP1_ave[:N]               # いらない後半部を除去
        CSP1_ave     = CSP1_ave/np.max(CSP1_ave)   # 最大で割り算

        # --ゼロ埋め込み周波数の決定----------
        pos         = np.argsort(-np.abs(CSP1))  # 振幅(周波数?)の大きい順にインデックスを取得
        embedded_freq   = pos[:D]                # CSPの最大D個の周波数

        # --CSP1(埋込周波数のみ)を求める----------
        CSP1_emb = np.zeros_like(CSP1)
        CSP1_emb[embedded_freq]     = CSP1[embedded_freq]
        CSP1_emb_ave     = irfft(CSP1_emb, axis=0)        # 逆STFT

        # CSPの整形
        CSP1_emb_ave     = CSP1_emb_ave[:N]               # いらない後半部を除去
        CSP1_emb_ave     = CSP1_emb_ave/np.max(CSP1_emb_ave)   # 最大で割り算

        # --振幅変調と位相変調----------
        # Y1 に対して振幅変調
        Y1emb[embedded_freq, :] = amp * Y1emb[embedded_freq, :]        # embedded_freqの周波数ビンにamp倍
        # Y1 に対して位相変調
        theta = emb_phase/180 * np.pi
        Y1emb[embedded_freq, :] = Y1emb[embedded_freq, :] * np.exp(1j*theta)
        # print(f'Y1emb shape: {Y1emb.shape}')

        # -音質検査用-----
        Y1zero[embedded_freq, k:k+3] = amp * Y1zero[embedded_freq, k:k+3]

        # --CSP2を求める----------
        Yspec   = Y1emb + Y2spec        # 埋め込み信号を利用している(Y1emb)
        XY2         = Yspec[:, k+K:k+2*K] * np.conj(Xspec[:, k+K:k+2*K])  # 相互相関(周波数領域)
        XY2abs       = np.abs(XY2)
        XY2abs[XY2abs < eps] = eps                # 分母がほぼ0になるのを防止
        CSP2_sp      = XY2 / XY2abs                  # 白色化相互相関(周波数領域)
        CSP2         = np.mean(CSP2_sp, axis=1)   # 時間方向で平均
        CSP2_ave     = irfft(CSP2, axis=0)        # 逆STFT

        # CSPの整形
        CSP2_ave     = CSP2_ave[:N]               # いらない後半部を除去
        CSP2_ave     = CSP2_ave/np.max(CSP2_ave)   # 最大で割り算

        # --CSP2(埋込周波数のみ)を求める----------
        CSP2_emb = np.zeros_like(CSP2)
        CSP2_emb[embedded_freq]     = CSP2[embedded_freq]
        CSP2_emb_ave = irfft(CSP2_emb, axis=0)  # 逆STFT

        # CSPの整形
        CSP2_emb_ave = CSP2_emb_ave[:N]  # いらない後半部を除去
        CSP2_emb_ave = CSP2_emb_ave / np.max(CSP2_emb_ave)  # 最大で割り算

        # --重み付き差分CSPを求める----------
        # -重みを計算する-----
        # CSPのピーク位置を計算
        pk_csp, _   = find_peaks(CSP1_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        index          = np.argsort(-CSP1_ave[pk_csp])
        # CSPの大きい順にD位のピーク位置をピークの大きい順に取得
        pk_csp      = pk_csp[index[:D]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1      = pk_csp[0]
        # # 重み (CSPから、CSPの最大ピークを除いた、D-1位のピークのみ抽出したもの)
        # weight      = np.zeros(CSP_ave.size)
        # weight[pk_csp[1:]] = CSP_ave[pk_csp[1:]]

        # 重み
        weight      = np.copy(CSP1_ave)
        weight[delay1-3:delay1+3] = 0    # 推定した第１スピーカのピークを除去
        weight[weight < TH]   = 0
        weight      = weight/np.max(np.abs(weight))  # 正規化


        '''---------------
            7. 重み付け差分CSPによる遅延推定
        ---------------'''
        # CSPの差分
        CSP_sub     = CSP1_ave - CSP2_ave     # CSPの差分
        CSP_sub     = CSP_sub / np.max(CSP_sub) # 正規化

        # 重み付け差分CSP
        CSP_wt      = weight*CSP_sub            # 重み付け埋め込み差分CSP

        '''---------------
            6.2 重み付け差分CSP(埋込周波数のみ)用の重み計算
        ---------------'''
        # CSPのピーク位置を計算
        pk_csp, _ = find_peaks(CSP1_emb_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        index = np.argsort(-CSP1_emb_ave[pk_csp])
        # CSPの大きい順にD位のピーク位置をピークの大きい順に取得
        pk_csp = pk_csp[index[:D]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1 = pk_csp[0]
        # # 重み (CSPから、CSPの最大ピークを除いた、D-1位のピークのみ抽出したもの)
        # weight      = np.zeros(CSP_ave.size)
        # weight[pk_csp[1:]] = CSP_ave[pk_csp[1:]]

        # 重み
        weight = np.copy(CSP1_emb_ave)
        weight[delay1 - 3:delay1 + 3] = 0  # 推定した第１スピーカのピークを除去
        weight[weight < TH] = 0
        weight = weight / np.max(np.abs(weight))  # 正規化

        '''---------------
            7. 重み付け差分CSP(埋込周波数のみ)による遅延推定
        ---------------'''
        # CSPの差分
        CSP_emb_sub     = CSP1_emb_ave - CSP2_emb_ave     # CSPの差分
        CSP_emb_sub     = CSP_emb_sub / np.max(CSP_emb_sub) # 正規化

        # 重み付け差分CSP
        CSP_emb_wt      = weight*CSP_emb_sub            # 重み付け埋め込み差分CSP

        # 重み付き差分CSPの図示
        # plt.figure()
        # plt.plot(CSP_ave, 'lightgray')
        # plt.plot(CSP_wt)
        # plt.show()

        # ログに溜め込む
        CSP_data.append(CSP1_ave)
        CSP_emb_data.append(CSP2_ave)
        CSP_sub_data.append(CSP_sub)
        CSP_wtd_data.append(CSP_wt)
        CSP1_data.append(CSP1_emb_ave)
        CSP2_data.append(CSP2_emb_ave)
        CSP_emb_sub_data.append(CSP_emb_sub)
        CSP_emb_wtd_data.append(CSP_emb_wt)


    # numpyに変更
    CSP_data     = np.array(CSP_data)         # CSP1
    CSP_emb_data = np.array(CSP_emb_data)     # CSP2
    CSP_sub_data = np.array(CSP_sub_data)     # 差分CSP
    CSP_wtd_data = np.array(CSP_wtd_data)     # 重み付き差分CSP
    CSP1_data     = np.array(CSP1_data)       # CSP1(埋込周波数のみ)
    CSP2_data     = np.array(CSP2_data)       # CSP2(埋込周波数のみ)
    CSP_emb_sub_data = np.array(CSP_emb_sub_data)     # 差分CSP
    CSP_emb_wtd_data = np.array(CSP_emb_wtd_data)     # 重み付き差分CSP

    # print('[ マイク・スピーカ距離・到来時間 ]')
    # print(f' - スピーカ1：{pos_imp[0]/fs*c:.2f}[m] , {1000*pos_imp[0]/fs:.2f}[ms]')
    # print(f' - スピーカ2：{pos_imp[1]/fs*c:.2f}[m] , {1000*pos_imp[1]/fs:.2f}[ms]\n')

    '''---------------
        遅延量推定精度
    ---------------'''
    Delay = []
    for csp1, csp2 in zip(CSP1_data, CSP_emb_wtd_data):
        # 遅延量の推定
        csp1_imp = []
        csp1_peaks, _ = find_peaks(csp1, height=0.7)
        csp1_imp.append(csp1_peaks[0])
        #delay1 = np.argmax(csp1)
        delay1 = csp1_imp[0]
        delay2 = np.argmax(csp2)
        # 遅延量をバッファリング
        delay = np.array([delay1, delay2])
        Delay.append(delay)

    Delay = np.array(Delay)

    # 遅延推定誤差 (平均絶対誤差)
    error = []
    for delay in Delay:
        tmp1 = np.sum(np.abs(delay - pos_imp_sub_d))
        tmp2 = np.sum(np.abs(np.flip(delay) - pos_imp_sub_d))
        error.append(np.min([tmp1, tmp2]))
    error = np.mean(np.array(error))
    delay_time_error = error / fs
    delay_time_error = 1000 * delay_time_error #[ms]に変換

    dte_data.append(delay_time_error)

    PR_data = []
    for csp2, delay in zip(CSP_emb_wtd_data, Delay):
        # まずcsp1が第１スピーカと第２スピーカどちらの遅延を検知したか判定
        if np.abs(delay[0] - pos_imp_sub_d[0]) < np.abs(delay[0] - pos_imp_sub_d[1]):
            pos_truth = pos_imp_sub_d[1]  # csp2はpos_imp[1]を推定したと判定
        else:
            pos_truth = pos_imp_sub_d[0]  # csp2はpos_imp[0]を推定したと判定

        # 真の遅延 pos_truth におけるピークの大きさ
        csp2_peak = csp2[pos_truth]

        # それ以外での最大ピーク
        tmp = np.copy(csp2)
        tmp[pos_truth] = 0
        peak_2nd = np.max(tmp)

        PR_data.append(csp2_peak / (np.max([peak_2nd, 10 ** (-8)])))

    PR_data = np.array(PR_data)

    # print('[ ピーク比(PR: Peak Ratio) ]')
    # print(f' - 平均PR: {np.mean(PR_data):.2f}')
    # print(f' - 最小PR: {np.min(PR_data):.2f}')
    # print(f' - 正しく検知できる確率: {PR_data[PR_data >= 1].size / PR_data.size:.3f}\n')


    '''---------------
        音質評価
    ---------------'''
    # 時間波形
    frames  = min([Y1spec.shape[1], Y1zero.shape[1]])
    y1_orig = istft(Y1spec[:,:frames], hop_length=S)
    y1_emb  = istft(Y1zero[:,:frames], hop_length=S)

    # PESQ
    y1_orig_ds = resample(y1_orig[:fs*5], orig_sr=fs, target_sr=fs)
    y1_emb_ds  = resample(y1_emb[:fs*5] , orig_sr=fs, target_sr=fs)
    score = pesq(16000, y1_orig_ds, y1_emb_ds)
    # SNR
    snr = 20 * np.log10(sum(y1_orig ** 2) / sum((y1_orig - y1_emb) ** 2))
    # print('[ 音質 ]')
    # print(f' - PESQ :  {score:.2f}')
    # print(f' - SNR  :  {snr:.2f} [dB]')

    pesq_data.append(score)

    sf.write(f'./../../sound_data/distance_estimation/music{music_type}_mono/embded_music{music_type}_gain={amp:.2f}.wav', y1_emb, fs)
    # 確認用の表示
    print(f'{(int(num+1) / loop_times)*100:3.0f}% Completed')

dte_data = np.array(dte_data)
# print(dte_data)
pesq_data = np.array(pesq_data)
# print(pesq_data)

# --埋込強度の変化に伴う推定誤差結果の変化をプロット----------
fig = plt.figure(num='埋込強度変化', figsize=(6, 3))
plt.subplots_adjust(bottom=0.15)
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(emb_amp, dte_data*c/1000, label='Distance Estimation Error')
ax1.set_xlim([-0.05,1.0])
#ax1.set_ylim([-0.05,1.75])
ax1.set_xlabel("Embedding Amplitude Gain")
ax1.set_ylabel("Estimation Distance Error[m]")

ax2 = ax1.twinx()
ax2.plot(emb_amp, pesq_data, 'r', label='PESQ')
ax2.set_ylim([-0.05, 5.5])
ax2.set_ylabel("PESQ")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc='lower right')

plt.savefig('./../../figures/distance_estimation/Amp_vs_PESQ.png')
# plt.show()