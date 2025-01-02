#!/bin/bash

# create_dir.sh
# Created by FreReRiku on 2025/01/03

# ルートディレクトリを指定
root_dir="."

# 必要なディレクトリ構造を配列で定義
dirs=(
    "$root_dir/data/distance_estimation"
    "$root_dir/data/room_simulation"
    "$root_dir/figure/distance_estimation"
    "$root_dir/figure/room_simulation"
    "$root_dir/sound/distance_estimation/music1_mono"
    "$root_dir/sound/distance_estimation/music2_mono"
    "$root_dir/sound/room_simulation"
)

# ディレクトリを一括作成
for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done
