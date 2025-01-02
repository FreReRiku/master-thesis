# MASTER-THESIS

このリポジトリは, 音響シミュレーション並びに信号処理を行うことを目的としています.

ソースコードはPythonで記述されています.

## はじめに

システムにPythonがインストールされていることを確認してください.

また, シミュレーションを行うために以下のライブラリをインストールしてください.

- `scipy`
- `numpy`
- `pyroomacoustics`
- `soundfile`
- `pesq`
- `librosa`
- `matplotlib`

これらはpipを使用してインストールできます.

例

```bash
pip install scipy numpy pyroomacoustics soundfile pesq librosa matplotlib
```

## ディレクトリ構造

- `src/` : 音声データの処理と分析に使用されるPythonソースコードが含まれています.
- `sound/` : `src/` ディレクトリのスクリプトで使用されるWAV形式の音声ファイルが格納されています.
- `doc/` : ソースコードに関するドキュメントが格納されています.

## スクリプトの実行

スクリプトを実行するには,srcディレクトリに移動し,Pythonスクリプトを実行します.

例

```bash
cd src/room_simulation
python main.py
```
