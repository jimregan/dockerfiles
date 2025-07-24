FROM kaldiasr/kaldi
WORKDIR /opt/kaldi/tools
RUN extras/install_irstlm.sh
WORKDIR /opt/kaldi/egs/sprakbanken_swe/s5
ENTRYPOINT ["/bin/bash", "-c", "run.sh"]
