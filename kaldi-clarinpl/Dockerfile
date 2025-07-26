FROM jimregan/kaldi_swe
WORKDIR /opt/kaldi/egs
RUN git clone https://github.com/danijel3/ClarinStudioKaldi
WORKDIR /opt/kaldi/egs/ClarinStudioKaldi
RUN apt update
RUN apt install -y libperlio-gzip-perl curl python-dev
COPY path.sh .
