#!/bin/bash

# ユーザー名を引数として受け取る
USER="kan"

# 引数が指定されていない場合のエラーチェック
if [ -z "$USER" ]; then
  echo "Usage: $0 <username>"
  exit 1
fi

# ユーザーが実行している Python プロセスを検索して終了
echo "Stopping all Python processes for user: $USER"

# 1. ユーザーのすべての python プロセスを停止
ps aux | grep python | grep $USER | awk '{print $2}' | xargs kill -9

echo "All Python processes for user $USER have been stopped."
