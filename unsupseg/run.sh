  docker run --rm --gpus all \
    --shm-size=4g \
    -e CUDA_LAUNCH_BLOCKING=1 \
    -v /home/joregan/waxholm-unsupseg:/data/waxholm-unsupseg:ro \
    -v /home/joregan/unsupseg-output:/output \
    jimregan/unsupseg \
    python main.py \
      data=buckeye \
      buckeye_path=/data/waxholm-unsupseg \
      buckeye_percent=1.0 \
      batch_size=1 \
      dataloader_n_workers=0 \
      gpus=1 \
      hydra.run.dir=/output/unsupseg-waxholm \
      exp_name=waxholm
