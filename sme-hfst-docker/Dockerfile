FROM ubuntu:23.04

WORKDIR /tmp

RUN apt update

RUN apt-get install -y cg3 hfst wget git autoconf subversion gawk icu-devtools make

RUN wget https://apertium.projectjj.com/apt/install-nightly.sh -O - | bash

RUN git clone https://github.com/giellalt/giella-core && \
    cd giella-core && \
    ./autogen.sh && \
    ./configure && \
    make

WORKDIR /tmp

ENV GIELLA_CORE=/tmp/giella-core

RUN git clone https://github.com/giellalt/lang-sme && \
    cd lang-sme && \
    ./autogen.sh && \
    ./configure && \
    make
