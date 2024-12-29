'''----------
convert.py
Created by FreReRiku on 2024/12/29
----------'''

import librosa
import soundfile as sf

def sampling_rate(input_file, orig_sr, target_sr):
    '''
    sampling_rate
    -----------

    音源を読み込み、サンプリングレートを変更したものを生成します。

    Parameter
    ----------
    input_file: 対象の音源
    orig_sr: 変換前のサンプリングレート
    target_sr: 変更後のサンプリングレート

    Return
    ----------
    この関数は何も返しません。
    '''

    wav, _ = librosa.load(f'{input_file}_{orig_sr}Hz.wav', sr=orig_sr)
    resampled_wav = librosa.resample(y=wav, orig_sr=orig_sr, target_sr=target_sr)
    sf.write(f'{input_file}_{target_sr}Hz.wav', resampled_wav, target_sr, 'PCM_16')
    return
