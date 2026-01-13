FROM lowerquality/gentle

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Put all Swedish model assets under /gentle/exp (Gentle's expected root)
ENV HF_BASE="https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main"

# A tiny helper that:
#  - downloads with wget (following redirects)
#  - fails if the content looks like HTML or a Git LFS pointer
#  - fails if the file is unexpectedly small
RUN cat >/usr/local/bin/hf_get <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

url="$1"
out="$2"
min_bytes="${3:-1}"

mkdir -p "$(dirname "$out")"

# -q: quiet, -O: output, --max-redirect: follow HF redirects robustly, --timeout/--tries: avoid hangs
wget -q --max-redirect=20 --timeout=30 --tries=3 -O "$out" "$url"

# size check
sz="$(stat -c%s "$out" 2>/dev/null || wc -c <"$out")"
if [[ "$sz" -lt "$min_bytes" ]]; then
  echo "ERROR: $out is too small ($sz bytes), expected >= $min_bytes"
  echo "URL: $url"
  exit 1
fi

# reject Git LFS pointer
if head -n 3 "$out" | grep -qE '^version https://git-lfs\.github\.com/spec/v1'; then
  echo "ERROR: $out is a Git LFS pointer, not the real file"
  echo "URL: $url"
  exit 1
fi

# reject HTML (common when you got a web page)
if head -n 5 "$out" | grep -qiE '<!doctype html|<html|<head|<body'; then
  echo "ERROR: $out looks like HTML, not a raw model blob"
  echo "URL: $url"
  exit 1
fi

# extra paranoia: for "binary" files, reject if it looks like plain UTF-8 text
# (Allows OpenFST text formats only if you set min_bytes small and skip this block.)
case "$out" in
  *.txt|*.conf|*splice_opts|*cmvn_opts|*/words.txt|*/phones.txt) : ;;
  *)
    # if the first 4KB are all printable/whitespace, it’s probably not a Kaldi/OpenFST binary
    if head -c 4096 "$out" | LC_ALL=C tr -d '\000\011\012\015\040-\176' | wc -c | grep -qx '0'; then
      echo "ERROR: $out looks like plain text; expected a binary model/FST"
      echo "URL: $url"
      exit 1
    fi
    ;;
esac

echo "OK: $out ($sz bytes)"
EOF
RUN chmod +x /usr/local/bin/hf_get

# Now fetch Swedish assets and *fail the build* if anything is wrong.
# IMPORTANT: use ?download=true to force blob delivery.
RUN mkdir -p /gentle/exp/tdnn_7b_chain_online/ivector_extractor && \
    mkdir -p /gentle/exp/langdir && \
    mkdir -p /gentle/exp/tdnn_7b_chain_online/graph_pp/phones && \
    \
    # Acoustic model (set minimum sizes that are "obviously real")
    hf_get "$HF_BASE/exp/chain/tdnn/final.mdl?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/final.mdl" 50000000 && \
    hf_get "$HF_BASE/exp/chain/tdnn/tree?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/tree" 500000 && \
    hf_get "$HF_BASE/exp/chain/tdnn/frame_subsampling_factor?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/frame_subsampling_factor" 1 && \
    \
    # i-vector extractor (use 10.ie as final.ie)
    hf_get "$HF_BASE/exp/chain/extractor/10.ie?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.ie" 1000000 && \
    hf_get "$HF_BASE/exp/chain/extractor/final.dubm?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.dubm" 100000 && \
    hf_get "$HF_BASE/exp/chain/extractor/final.mat?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.mat" 10000 && \
    hf_get "$HF_BASE/exp/chain/extractor/global_cmvn.stats?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/global_cmvn.stats" 1000 && \
    hf_get "$HF_BASE/exp/chain/extractor/splice_opts?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/splice_opts" 10 && \
    hf_get "$HF_BASE/exp/chain/extractor/online_cmvn.conf?download=true" \
      "/gentle/exp/tdnn_7b_chain_online/ivector_extractor/online_cmvn.conf" 1 && \
    \
    # langdir (you said you have lang_chain locally; once it's on HF, point HF_BASE at it)
    # Example paths — adjust to whatever you upload:
    hf_get "$HF_BASE/data/lang_chain/L.fst?download=true" \
      "/gentle/exp/langdir/L.fst" 1000 && \
    hf_get "$HF_BASE/data/lang_chain/L_disambig.fst?download=true" \
      "/gentle/exp/langdir/L_disambig.fst" 1000 && \
    hf_get "$HF_BASE/data/lang_chain/phones.txt?download=true" \
      "/gentle/exp/langdir/phones.txt" 100 && \
    hf_get "$HF_BASE/data/lang_chain/words.txt?download=true" \
      "/gentle/exp/langdir/words.txt" 100 && \
    hf_get "$HF_BASE/data/lang_chain/topo?download=true" \
      "/gentle/exp/langdir/topo" 10 && \
    hf_get "$HF_BASE/data/lang_chain/oov.int?download=true" \
      "/gentle/exp/langdir/oov.int" 1 && \
    hf_get "$HF_BASE/data/lang_chain/oov.txt?download=true" \
      "/gentle/exp/langdir/oov.txt" 1 && \
    \
    # "graph_pp" metadata: if Gentle insists on these paths existing,
    # mirror phones/words/phones.txt from langdir (all small text, so checks are light).
    cp -a /gentle/exp/langdir/words.txt  /gentle/exp/tdnn_7b_chain_online/graph_pp/words.txt && \
    cp -a /gentle/exp/langdir/phones.txt /gentle/exp/tdnn_7b_chain_online/graph_pp/phones.txt && \
    cp -a /gentle/exp/langdir/phones/*   /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/ || true

