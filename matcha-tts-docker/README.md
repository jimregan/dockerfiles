# Matcha-TTS local Docker image

Self-contained build of the [kth-tmh/Matcha-TTS](https://huggingface.co/spaces/kth-tmh/Matcha-TTS)
Gradio Space, with the four checkpoints baked into the image so the container
needs no network access at runtime.

## Build

```bash
docker build -t matcha-tts .
```

The checkpoints (~hundreds of MB) are downloaded once during the build from the
[Matcha-TTS-checkpoints](https://github.com/shivammehta25/Matcha-TTS-checkpoints)
GitHub release and stored under `/data/matcha_tts` (`MATCHA_HOME`).

## Run

```bash
docker run --rm -p 7860:7860 matcha-tts
```

Then open http://localhost:7860

`GRADIO_SERVER_NAME=0.0.0.0` is set in the image so Gradio binds to all
interfaces and is reachable from the host (the Space's `app.py` calls
`launch()` with no args, which would otherwise bind to localhost inside the
container).

## Notes

- Built on Python 3.10 deliberately: the Space's default builder moved to
  Python 3.13, where `matcha-tts`'s pinned build deps (`numpy==1.24.3`,
  `cython==0.29.35`) have no wheels and fail to compile.
- CPU inference works out of the box. For GPU, add an NVIDIA base image / CUDA
  torch and run with `--gpus all`.
