#!/usr/bin/env sh
set -eu

if [ "$#" -gt 0 ]; then
  exec ctc-forced-aligner "$@"
fi

audio_path="${AUDIO_PATH:-}"
text_path="${TEXT_PATH:-}"

if [ -z "$audio_path" ]; then
  audio_path="$(find /input -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.ogg' \) | sort | head -n 1)"
fi

if [ -z "$text_path" ] && [ -n "$audio_path" ]; then
  base="$(basename "$audio_path")"
  stem="${base%.*}"
  if [ -f "/input/$stem.txt" ]; then
    text_path="/input/$stem.txt"
  fi
fi

if [ -z "$text_path" ]; then
  text_path="$(find /input -maxdepth 1 -type f -iname '*.txt' | sort | head -n 1)"
fi

if [ -z "$audio_path" ] || [ -z "$text_path" ]; then
  printf '%s\n' "Expected an audio file and a .txt transcript in /input, or pass ctc-forced-aligner arguments explicitly." >&2
  exit 2
fi

set -- \
  --audio_path "$audio_path" \
  --text_path "$text_path" \
  --language "${LANGUAGE:-eng}" \
  --split_size "${SPLIT_SIZE:-word}" \
  --device "${DEVICE:-cpu}"

if [ "${ROMANIZE:-1}" = "1" ]; then
  set -- "$@" --romanize
fi

if [ -n "${ALIGNMENT_MODEL:-}" ]; then
  set -- "$@" --alignment_model "$ALIGNMENT_MODEL"
fi

exec ctc-forced-aligner "$@"

