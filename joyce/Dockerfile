FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y libsdl2-dev wget automake libxml2-dev libpng-dev \
                       build-essential
RUN wget https://www.seasip.info/Unix/Joyce/joyce-2.5.4.tar.gz
RUN tar zxvf joyce-2.5.4.tar.gz
WORKDIR joyce-2.5.4
RUN ./configure
RUN make && make install
