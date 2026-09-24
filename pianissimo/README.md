# Pianissimo in Docker

NVIDIA NeMo container for
[`KlangAI/pianissimo-sv`](https://huggingface.co/KlangAI/pianissimo-sv).

The target Linux host needs Docker and the NVIDIA Container Toolkit. The image
uses CUDA 12.6 and is compatible with NVIDIA driver 560.35.05. NeMo 2.5 is
installed on top of the CUDA 12.6 PyTorch runtime.

The model checkpoint is downloaded into the image at build time and pinned to
Hugging Face revision `8f1f6d8f8bd7482a5ea1d2bfaf6ef5be61597138`.

```sh
docker build -t pianissimo-sv .
```

To deliberately build another model revision:

```sh
docker build \
  --build-arg MODEL_REVISION=<full-hugging-face-commit> \
  -t pianissimo-sv .
```

## Batch transcription on eight GPUs

The batch runner starts one persistent process per visible GPU. Each process
loads its own model once, while the parent process scans the input tree and
groups similarly sized recordings into batches. Output paths mirror the input
tree, replacing `.wav` with `.json`.

```sh
docker run --rm --gpus all --ipc=host \
  -v /absolute/path/to/wavs:/input:ro \
  -v /absolute/path/to/json:/output \
  pianissimo-sv batch /input /output \
  --batch-size 16 \
  --timestamps word-segment
```

Existing JSON outputs are skipped, so the same command resumes an interrupted
run. Writes are atomic. A failed input produces an adjacent `.error.json` and
the process exits nonzero after the remaining work finishes.

For 30-second files on 24 GB RTX 3090 cards, start at batch size 16. Watch GPU
memory and utilization with `nvidia-smi`; try 24 and then 32 if memory permits.
Duration bucketing limits padding waste. If timestamps are unnecessary,
`--timestamps none` reduces output size and some processing overhead. Use
`--timestamps word` when segment timestamps are not needed.

The scanner streams directory entries and does not hold the complete filename
list in memory, including when millions of WAV files occupy one directory. A
flat output directory works too, although its filesystem metadata operations
may be slower than a sharded layout. Put input and output on storage that can
sustain concurrent reads and metadata writes.

## Single-file transcription

```sh
docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  pianissimo-sv /audio/audio.wav
```

The model does not download at runtime. Input should be mono, 16 kHz audio.

For word and segment timestamps:

```sh
docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  pianissimo-sv --timestamps /audio/audio.wav
```

For several short files, list them and optionally raise the batch size:

```sh
docker run --rm --gpus all \
  -v "$PWD:/audio:ro" \
  pianissimo-sv --batch-size 8 /audio/one.wav /audio/two.wav
```

Use batch size 1 for long recordings. Both runners accept `--model` to override
the checkpoint embedded at `/models/pianissimo-sv.nemo`.
