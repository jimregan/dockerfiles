# Pianissimo in Docker

Quick NVIDIA NeMo container for
[`KlangAI/pianissimo-sv`](https://huggingface.co/KlangAI/pianissimo-sv).

The target Linux host needs Docker and the NVIDIA Container Toolkit. The image
uses CUDA 12.6 and is compatible with NVIDIA driver 560.35.05. NeMo 2.5 is
installed on top of the CUDA 12.6 PyTorch runtime rather than using NVIDIA's
CUDA 12.9 NeMo image.

```sh
docker build -t pianissimo .

docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  -v pianissimo-hf-cache:/cache \
  pianissimo /audio/audio.wav
```

The first run downloads the roughly 2.5 GB model. Output is one JSON object per
input file. The input should be mono, 16 kHz audio; NeMo/FFmpeg can handle common
audio formats, but WAV is the least surprising option.

For word and segment timestamps:

```sh
docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  -v pianissimo-hf-cache:/cache \
  pianissimo --timestamps /audio/audio.wav
```

For several short files, list them and optionally raise the batch size:

```sh
docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  -v pianissimo-hf-cache:/cache \
  pianissimo --batch-size 8 /audio/one.wav /audio/two.wav
```

Use batch size 1 for long recordings. Add `--model /models/pianissimo-sv.nemo`
and mount a downloaded checkpoint at `/models` if the container must run
offline.
