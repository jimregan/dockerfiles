#!/bin/bash

mkdir -p /data/smplx_npz
mkdir -p /data/logs

GPUS=(0 1 2 3 4 5 6 7)   # List of available GPU IDs
NUM_GPUS=${#GPUS[@]}
# FILES=("/home/deichler/data/sgs_recordings/hsi/pos4smplx_dataset/"*.npy)
FILES=(/data/*.npy) # start with referential

# Split files into batches for each GPU
declare -A GPU_FILES

for i in "${!FILES[@]}"; do
  gpu_idx=$(( i % NUM_GPUS ))
  gpu_id=${GPUS[$gpu_idx]}
  GPU_FILES[$gpu_id]="${GPU_FILES[$gpu_id]} ${FILES[$i]}"
done

MAX_PARALLEL=6  # how many parallel jobs PER GPU

# Process files assigned to each GPU
for gpu_id in "${GPUS[@]}"; do
  (
    file_list=${GPU_FILES[$gpu_id]}
    count=0
    for npy_file in $file_list; do
      base_name=$(basename "$npy_file" .npy)
      echo "Starting $base_name on GPU $gpu_id..."
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py --npy "$npy_file" \
        --shape_pkl ./test_data/P2.pkl \
        --save_path "/data/smplx_npz/${base_name}_smplx.npz" \
        > "/data/logs/${base_name}.log" 2>&1 &
      
      ((count++))
      
      # If we reach MAX_PARALLEL jobs, wait for them to finish before starting more
      if (( count % MAX_PARALLEL == 0 )); then
        wait
      fi
    done
    wait
  ) &
done

wait
echo "All jobs completed!"
