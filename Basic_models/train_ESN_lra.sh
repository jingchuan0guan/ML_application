#!/bin/bash

# 引数の設定
sizes=(32 64 128 256)  # サイズのリスト
types=("E" "M" "H")  # タイプのリスト

# ループしてコマンド実行
for size in "${sizes[@]}"; do
  for type in "${types[@]}"; do
    log_file="log/log_${size}_${type}.out"
    
    # nohupを使ってバックグラウンドで実行し、出力をログファイルに書き込む
    echo "Running with size $size and type $type. Output will be logged in $log_file"
    nohup python train_ESN_lra.py $size $type > $log_file 2>&1 &
  done
done

echo "All jobs are started."