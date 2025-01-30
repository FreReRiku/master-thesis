# MASTER-THESIS

本プロジェクトは、**音響シミュレーション** を用いて **GCC-PHAT 法による距離推定** を行う Python スクリプト群です。`pyroomacoustics` を用いた音場シミュレーションと、`GCC-PHAT` による距離推定アルゴリズムを実装しています。

## 1. ディレクトリ構成

```bash
.
├── Makefile
├── README.md
├── requirements.txt
├── data
│   ├── distance_estimation
│   │   ├── music1_mono
│   │   │   └── csv_files
│   │   │       ├── logs
│   │   │       └── raw_data
│   │   │  
│   │   └── music2_mono
│   │       └── csv_files
│   │           ├── logs
│   │           └── raw_data
│   │
│   └── room_simulation
│       ├── room_info
│       └── sr_and_spk
│
├── figure
│   ├── distance_estimation
│   │   ├── music1_mono
│   │   └── music2_mono
│   └── room_simulation
│
├── sound
│   ├── distance_estimation
│   │   ├── music1_mono
│   │   └── music2_mono
│   ├── original
│   │   ├── music1_mono.wav
│   │   └── music2_mono.wav
│   └── room_simulation
├── src
│   ├── distance_estimation
│   │   ├── calc.py
│   │   ├── main.py
│   │   ├── save.py
│   │   └── visualize.py
│   └── room_simulation
│       ├── convert.py
│       ├── main.py
│       ├── save.py
│       └── simulate.py
│
└── .gitignore
```

## 2. インストール方法

本プロジェクトは Python で動作します。以下の手順で必要な環境をセットアップしてください。

### 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 3. 使用方法

### 全体の処理を実行する

```bash
make
```

このコマンドで **部屋の音響シミュレーションと距離推定の処理をすべて実行** できます。

### 個別に実行する

#### 1. 音響シミュレーションのみ実行

```bash
make room_simulation
```

#### 2. 距離推定のみ実行

```bash
make distance_estimation
```

## 4. 処理の流れ

1. **`room_simulation/main.py` の実行**
    - `simulate.py` により音場のシミュレーションを行い、 `sound/room_simulation/` にデータを保存。
2. **`distance_estimation/main.py` の実行**
    - `calc.py` によりGCC-PHATを用いた遅延推定を行い、 `data/distance_estimation/musicX_mono/csv_files/raw_data/` にデータを保存。
    - `visualize.py` によりデータを可視化し、 `figure/distance_estimation/musicX_mono/` にグラフを保存。

## 5. スクリプトの説明

### 1. `room_simulation` の処理

- `main.py` ： `simulate.py` を呼び出して音響シミュレーションを実行。
- `simulate.py` ： `pyroomacoustics` を用いて仮想空間内の音の伝播を計算。

### 2. `distance_estimation` の処理

- `main.py` ： `calc.py` と `visualize.py` を順番に実行。
- `calc.py` ： `GCC-PHAT` 法を用いた遅延推定を行い、結果を CSV に保存。
- `visualize.py` ： 計算結果をグラフ化し、 `figure/` に保存。
- `save.py` ： 計算結果の CSV 出力を担当。

## 6. 出力ファイル

### 1. 音響シミュレーションの出力

- `sound/room_simulation/` ： シミュレーション結果の音源データ。
- `figure/room_simulation/room.png` ： シミュレーション環境の可視化画像。

### 2. 距離推定の出力

- `data/distance_estimation/musicX_mono/csv_files/raw_data/` ： 計算結果の CSV ファイル。
- `figure/distance_estimation/musicX_mono/` ： 可視化結果のグラフ（SVG 形式）。

## 7. 注意事項

- `sound/original/` には、解析対象の音源ファイル（`music1_mono.wav`, `music2_mono.wav` など）を配置してください。
- `.gitignore` により、生成されるデータ（`figure/`, `sound/distance_estimation/`, `data/` など）は Git の管理対象外となっています。
- `make` 実行時にディレクトリが自動生成されるため、手動で `mkdir` する必要はありません。
