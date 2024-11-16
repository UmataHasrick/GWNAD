#!/bin/bash

# 检查参数个数
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <min_idx> <max_idx> <device_mode> <(suffix_for_output)>"
  exit 1
fi

# 获取参数
min_idx=$1
max_idx=$2
device=$3
suffix=$4

# 计算差值是否为8的倍数
diff=$((max_idx - min_idx))
if [ $((diff % 8)) -ne 0 ]; then
  echo "Error: Difference between min_idx and max_idx must be a multiple of 8."
  exit 1
fi

# 计算每个screen的idx
step=$((diff / 8))

# 创建16个screen并执行命令
for i in $(seq 0 7); do
  screen_idx=$((min_idx + i * step))
  screen_idx_upperbound=$((min_idx + i * step + step))
  
  if [ $i -lt 4 ]; then
    param=1
  else
    param=2
  fi

  if [ $device -eq 0 ]; then
    param=0
  fi

  screen -dmS screen_$i sh -c "echo 'executing command for screen_$i with idx=$screen_idx'; python3 SeriesWSC_AEalone_suffix_GWAK.py $param $screen_idx $screen_idx_upperbound $suffix"

  echo 'executing command for screen_$i with idx=$screen_idx'
  echo $suffix

done
