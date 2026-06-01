FROM debian:jessie-20171009

RUN echo "deb http://archive.debian.org/debian jessie main" > /etc/apt/sources.list && \
     echo "deb http://archive.debian.org/debian-security jessie/updates main" >> /etc/apt/sources.list && \
     apt-get -o Acquire::Check-Valid-Until=false update && \
     apt-get -o Acquire::Check-Valid-Until=false install -y --force-yes cvs2svn wget unzip cvs git

WORKDIR /workspace
