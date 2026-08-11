  docker run --rm --gpus all \
    -v /home/joregan/waxholm-unsupseg:/data/waxholm-unsupseg:ro \
    -v /home/joregan/unsupseg-output:/output \
    jimregan/unsupseg \
    python main.py \
      data=buckeye \
      buckeye_path=/data/waxholm-unsupseg \
      buckeye_percent=1.0 \
      hydra.run.dir=/output/unsupseg-waxholm \
      exp_name=waxholm
