FROM jimregan/kaldi_swe
RUN apt update
RUN apt install -y libperlio-gzip-perl curl python-dev-is-python3
WORKDIR /opt/kaldi/egs
RUN git clone https://github.com/danijel3/ClarinStudioKaldi
WORKDIR /opt/kaldi/tools
COPY install_phonetisaurus.sh /opt/kaldi/tools/extras/
RUN extras/install_phonetisaurus.sh
WORKDIR /opt/kaldi/egs/ClarinStudioKaldi
COPY path.sh .
COPY run.sh .
