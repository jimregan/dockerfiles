#!/usr/bin/env bash

stage=0
train=true
decode=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

mfccdir=mfcc

if [ $stage -le 0 ]; then
  echo "Stage 0: Preparing data..."
  local/sprak_data_prep.sh || exit 1
  utils/fix_data_dir.sh data/train || exit 1
  utils/fix_data_dir.sh data/test || exit 1
fi

# Dictionary & Lang prep
if [ $stage -le 1 ]; then
  echo "Stage 1: Preparing dictionary and language model..."

  local/copy_dict.sh || exit 1

  # Clean up dictionary: remove single and double quotes from pronunciations
  echo "Fixing dictionary (removing quotes)..."
  awk '{a=$1; gsub("\047", ""); gsub("\042", ""); $1=a; print}' \
    data/local/dict/lexicon.txt > data/local/dict/lexicon.fixed.txt
  mv data/local/dict/lexicon.fixed.txt data/local/dict/lexicon.txt

  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang || exit 1

  local/train_irstlm.sh data/local/transcript_lm/transcripts.uniq 4 "4g" data/lang data/local/train4_lm &> data/local/4g.log || exit 1
fi

# Feature extraction
if [ $stage -le 2 ]; then
  echo "Stage 2: Extracting MFCCs and CMVN..."
  for x in train test; do
    steps/make_mfcc.sh --nj 10 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir || exit 1
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1
    utils/fix_data_dir.sh data/$x || exit 1
  done
fi

# Monophone
if [ $stage -le 3 ] && $train; then
  echo "Stage 3: Training monophone model..."
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" data/train data/lang exp/mono || exit 1
fi

if [ $stage -le 3 ] && $decode; then
  echo "Stage 3: Decoding with monophone model..."
  utils/mkgraph.sh data/lang_test_4g exp/mono exp/mono/graph_4g || exit 1
  steps/decode.sh --config conf/decode.config --nj 10 --cmd "$decode_cmd" \
    exp/mono/graph_4g data/test exp/mono/decode_test || exit 1
fi

# Triphone1 (Deltas)
if [ $stage -le 4 ] && $train; then
  echo "Stage 4: Training tri1..."
  steps/align_si.sh --nj 10 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1
  steps/train_deltas.sh --cmd "$train_cmd" 5800 96000 data/train data/lang exp/mono_ali exp/tri1 || exit 1
fi

if [ $stage -le 4 ] && $decode; then
  echo "Stage 4: Decoding with tri1..."
  utils/mkgraph.sh data/lang_test_4g exp/tri1 exp/tri1/graph_4g || exit 1
  steps/decode.sh --config conf/decode.config --nj 10 --cmd "$decode_cmd" \
    exp/tri1/graph_4g data/test exp/tri1/decode_test || exit 1
fi

# Triphone2 (LDA+MLLT)
if [ $stage -le 5 ] && $train; then
  echo "Stage 5: Training tri2b (LDA+MLLT)..."
  steps/align_si.sh --nj 10 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali || exit 1
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=5 --right-context=5" \
    7500 125000 data/train data/lang exp/tri1_ali exp/tri2b || exit 1
fi

if [ $stage -le 5 ] && $decode; then
  echo "Stage 5: Decoding with tri2b..."
  utils/mkgraph.sh data/lang_test_4g exp/tri2b exp/tri2b/graph_4g || exit 1
  steps/decode.sh --nj 10 --cmd "$decode_cmd" \
    exp/tri2b/graph_4g data/test exp/tri2b/decode_test || exit 1
fi

# Triphone3 (SAT)
if [ $stage -le 6 ] && $train; then
  echo "Stage 6: Training tri3b (SAT)..."
  steps/align_si.sh --nj 10 --cmd "$train_cmd" data/train data/lang exp/tri2b exp/tri2b_ali || exit 1
  steps/train_sat.sh --cmd "$train_cmd" 7500 125000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1
fi

if [ $stage -le 6 ] && $decode; then
  echo "Stage 6: Decoding with tri3b..."
  utils/mkgraph.sh data/lang_test_4g exp/tri3b exp/tri3b/graph_4g || exit 1
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
    exp/tri3b/graph_4g data/test exp/tri3b/decode_test || exit 1
fi

# Triphone4 (Final SAT)
if [ $stage -le 7 ] && $train; then
  echo "Stage 7: Training tri4a (final SAT)..."
  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" data/train data/lang exp/tri3b exp/tri3b_ali || exit 1
  steps/train_sat.sh --cmd "$train_cmd" 13000 300000 data/train data/lang exp/tri3b_ali exp/tri4a || exit 1
fi

if [ $stage -le 7 ] && $decode; then
  echo "Stage 7: Decoding with tri4a..."
  utils/mkgraph.sh data/lang_test_4g exp/tri4a exp/tri4a/graph_4g || exit 1
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 \
    exp/tri4a/graph_4g data/test exp/tri4a/decode_test || exit 1
fi

# TDNN (Vosk-style)
if [ $stage -le 8 ] && $train; then
  echo "Stage 8: Training TDNN using chain model (Vosk-style)..."
  local/chain/run_tdnn.sh --train-set train --gmm tri4a || exit 1
fi

# Decode
if [ $stage -le 9 ] && $decode; then
  echo "Stage 9: Decoding with TDNN..."

  # Format LM and build decoding graph if needed
  utils/format_lm.sh data/lang data/local/train4_lm/lm_tgsmall.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test_4g || exit 1

  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_4g \
    exp/chain/tdnn exp/chain/tdnn/graph || exit 1

  # Extract MFCCs for test again if needed
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/test exp/make_mfcc/test $mfccdir
  steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir

  # Extract ivectors for test
  steps/online/nnet2/extract_ivectors_online.sh --nj 10 \
    data/test exp/chain/extractor exp/chain/ivectors_test || exit 1

  # Decode
  steps/nnet3/decode.sh --cmd "$decode_cmd" --nj 10 \
    --online-ivector-dir exp/chain/ivectors_test \
    exp/chain/tdnn/graph data/test exp/chain/tdnn/decode_test || exit 1

  # Rescore
  utils/build_const_arpa_lm.sh data/local/train4_lm/lm_tgmed.arpa.gz data/lang data/lang_test_rescore
  steps/lmrescore_const_arpa.sh data/lang_test_4g data/lang_test_rescore \
    data/test exp/chain/tdnn/decode_test exp/chain/tdnn/decode_test_rescore
fi

# WER summary
if [ $stage -le 10 ]; then
  echo "Stage 10: Summary of WERs..."
  for x in exp/*/decode*; do
    [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
  done
fi

