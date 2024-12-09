# MASTER-THESIS

このリポジトリは,音響データの研究及び分析を目的としています.

リポジトリ内には,ソースコードが入ったディレクトリと関連する音源データが入ったディレクトリが含まれています.

ソースコードはPythonで記述されています.

## ディレクトリ構造

- `src/` : 音声データの処理と分析に使用されるPythonソースコードが含まれています.
- `sound_data/` : `src/` ディレクトリのスクリプトで使用されるWAV形式の音声ファイルが格納されています.
- `doc/` : ソースコードに関するドキュメントが格納されています.

## はじめに

### 前提条件

はじめにシステムにPythonがインストールされていることを確認してください.

加えて,このディレクトリでは
- `scipy`
- `numpy`
- `matplotlib`
- `pyroomacoustics`
- `librosa`
- `soundfile`

を使用します.

これらはpipを使用してインストールできます.

```bash
pip install scipy numpy matplotlib pyroomacoustics pyroomacoustics librosa soundfile
```

#### スクリプトの実行

スクリプトを実行するには,src/ ディレクトリに移動し,Pythonスクリプトを実行します.
例えば,

```bash
cd src
python script_name.py #各自,実行したいスクリプトの名前に置き換えてください.
```

#### 音源データ

音源データを解析に用いる場合は,スクリプト内のパスがこのディレクトリ内のデータファイルを正しく指していることを確認してください.

