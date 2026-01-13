FROM lowerquality/gentle

RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/extractor/10.ie?download=true -O /gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.ie
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/extractor/global_cmvn.stats?download=true -O /gentle/exp/tdnn_7b_chain_online/ivector_extractor/global_cmvn.stats
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/extractor/splice_opts?download=true -O /gentle/exp/tdnn_7b_chain_online/ivector_extractor/splice_opts
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/extractor/final.mat?download=true -O /gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.mat
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/extractor/final.dubm?download=true -O /gentle/exp/tdnn_7b_chain_online/ivector_extractor/final.dubm

RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/L.fst?download=true -O /gentle/exp/langdir/L.fst
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/L_disambig.fst?download=true -O /gentle/exp/langdir/L_disambig.fst
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/oov.int?download=true -O /gentle/exp/langdir/oov.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/oov.txt?download=true -O /gentle/exp/langdir/oov.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones.txt?download=true -O /gentle/exp/langdir/phones.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/align_lexicon.int?download=true -O /gentle/exp/langdir/phones/align_lexicon.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/align_lexicon.txt?download=true -O /gentle/exp/langdir/phones/align_lexicon.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/context_indep.csl?download=true -O /gentle/exp/langdir/phones/context_indep.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/context_indep.int?download=true -O /gentle/exp/langdir/phones/context_indep.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/context_indep.txt?download=true -O /gentle/exp/langdir/phones/context_indep.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/disambig.csl?download=true -O /gentle/exp/langdir/phones/disambig.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/disambig.int?download=true -O /gentle/exp/langdir/phones/disambig.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/disambig.txt?download=true -O /gentle/exp/langdir/phones/disambig.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/extra_questions.int?download=true -O /gentle/exp/langdir/phones/extra_questions.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/extra_questions.txt?download=true -O /gentle/exp/langdir/phones/extra_questions.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/nonsilence.csl?download=true -O /gentle/exp/langdir/phones/nonsilence.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/nonsilence.int?download=true -O /gentle/exp/langdir/phones/nonsilence.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/nonsilence.txt?download=true -O /gentle/exp/langdir/phones/nonsilence.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.csl?download=true -O /gentle/exp/langdir/phones/optional_silence.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.int?download=true -O /gentle/exp/langdir/phones/optional_silence.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.txt?download=true -O /gentle/exp/langdir/phones/optional_silence.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/roots.int?download=true -O /gentle/exp/langdir/phones/roots.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/roots.txt?download=true -O /gentle/exp/langdir/phones/roots.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/sets.int?download=true -O /gentle/exp/langdir/phones/sets.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/sets.txt?download=true -O /gentle/exp/langdir/phones/sets.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/silence.csl?download=true -O /gentle/exp/langdir/phones/silence.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/silence.int?download=true -O /gentle/exp/langdir/phones/silence.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/silence.txt?download=true -O /gentle/exp/langdir/phones/silence.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/wdisambig.txt?download=true -O /gentle/exp/langdir/phones/wdisambig.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/wdisambig_phones.int?download=true -O /gentle/exp/langdir/phones/wdisambig_phones.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/wdisambig_words.int?download=true -O /gentle/exp/langdir/phones/wdisambig_words.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/word_boundary.int?download=true -O /gentle/exp/langdir/phones/word_boundary.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/word_boundary.txt?download=true -O /gentle/exp/langdir/phones/word_boundary.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/topo?download=true -O /gentle/exp/langdir/topo
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/words.txt?download=true -O /gentle/exp/langdir/words.txt

RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/tdnn/0.mdl?download=true -O /gentle/exp/tdnn_7b_chain_online/final.mdl

RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/tdnn/tree?download=true -O /gentle/exp/tdnn_7b_chain_online/tree
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/exp/chain/tdnn/phones.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/phones.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/raw/main/exp/chain/tdnn/frame_subsampling_factor?download=true -O /gentle/exp/tdnn_7b_chain_online/frame_subsampling_factor

RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/align_lexicon.int?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/align_lexicon.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/disambig.int?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/disambig.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/disambig.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/disambig.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.csl?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/optional_silence.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.int?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/optional_silence.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/optional_silence.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/optional_silence.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/silence.csl?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/silence.csl
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/word_boundary.int?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/phones/word_boundary.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.txt
RUN wget -q https://huggingface.co/jimregan/kaldi-sprakbanken_swe/resolve/main/data/lang/words.txt?download=true -O /gentle/exp/tdnn_7b_chain_online/graph_pp/words.txt

