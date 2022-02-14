#!/bin/bash

# Script to reproduce results

for ((i = 0; i < 3; i += 1)); do
  python main.py \
    --policy "TD3" \
    --env "HalfCheetah-v3" \
    --gpu_id 0 \
    --seed $i

  python main.py \
    --policy "TD3" \
    --env "Hopper-v3" \
    --gpu_id 1 \
    --seed $i

  python main.py \
    --policy "TD3" \
    --env "Walker2d-v3" \
    --gpu_id 2 \
    --seed $i

  python main.py \
    --policy "TD3" \
    --env "Ant-v3" \
    --gpu_id 3 \
    --seed $i

done

python main.py \
  --policy "TD3" \
  --env "Humanoid-v3" \
  --gpu_id 0 \
  --seed 0

python main.py \
  --policy "TD3" \
  --env "Humanoid-v3" \
  --gpu_id 1 \
  --seed 1

python main.py \
  --policy "TD3" \
  --env "Humanoid-v3" \
  --gpu_id 2 \
  --seed 2
