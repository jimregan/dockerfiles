#!/usr/bin/env sh
set -eu

if [ "$#" -gt 0 ]; then
  exec ctc-forced-aligner "$@"
fi

audio_list="$(mktemp)"
find /input -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.ogg' \) | sort > "$audio_list"

if [ ! -s "$audio_list" ]; then
  rm -f "$audio_list"
  printf '%s\n' "No audio files found under /input." >&2
  exit 2
fi

status=0

while IFS= read -r audio_path; do
  rel_path="${audio_path#/input/}"
  rel_stem="${rel_path%.*}"
  text_path="/input/$rel_stem.txt"
  output_path="/output/$rel_stem.json"

  if [ ! -f "$text_path" ]; then
    printf '%s\n' "Skipping $audio_path: missing $text_path" >&2
    continue
  fi

  work_dir="$(mktemp -d)"
  work_audio="$work_dir/audio.${audio_path##*.}"
  work_text="$work_dir/transcript.txt"
  cp "$audio_path" "$work_audio"
  cp "$text_path" "$work_text"

  set -- \
    --audio_path "$work_audio" \
    --text_path "$work_text" \
    --language "${LANGUAGE:-eng}" \
    --split_size "${SPLIT_SIZE:-word}" \
    --device "${DEVICE:-cpu}"

  if [ "${ROMANIZE:-1}" = "1" ]; then
    set -- "$@" --romanize
  fi

  if [ -n "${ALIGNMENT_MODEL:-}" ]; then
    set -- "$@" --alignment_model "$ALIGNMENT_MODEL"
  fi

  if ctc-forced-aligner "$@"; then
    mkdir -p "$(dirname "$output_path")"
    cp "${work_audio%.*}.json" "$output_path"
    printf '%s\n' "Wrote $output_path"
  else
    status=1
  fi

  rm -rf "$work_dir"
done < "$audio_list"

rm -f "$audio_list"
exit "$status"

