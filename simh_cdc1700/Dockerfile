# syntax=docker/dockerfile:1

ARG DEBIAN_VERSION=bookworm
ARG SIMH_REF=master
ARG MSOS_TAPE_URL=https://bitsavers.org/bits/CDC/1700_Cyber18/20100524/MSOS5_SL136.tap

FROM debian:${DEBIAN_VERSION}-slim AS build

ARG SIMH_REF
ARG MSOS_TAPE_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/simh
RUN git clone --depth 1 --branch "${SIMH_REF}" https://github.com/open-simh/simh.git .
RUN make cdc1700 NONETWORK=1 NOVIDEO=1 TESTS=0
RUN wget --progress=dot:giga --output-document /src/MSOS5_SL136.tap "${MSOS_TAPE_URL}"

FROM debian:${DEBIAN_VERSION}-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/simh/BIN/cdc1700 /usr/local/bin/cdc1700
COPY --from=build /src/simh/CDC1700/msosInstall.simh /opt/cdc1700/msosInstall.simh
COPY --from=build /src/simh/CDC1700/msosRun.simh /opt/cdc1700/msosRun.simh
COPY --from=build /src/simh/CDC1700/CDC1700-MSOS.txt /opt/cdc1700/CDC1700-MSOS.txt
COPY --from=build /src/MSOS5_SL136.tap /opt/cdc1700/MSOS5_SL136.tap

RUN chmod +x /usr/local/bin/cdc1700

WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["cdc1700"]
CMD ["/opt/cdc1700/msosInstall.simh", "/opt/cdc1700/MSOS5_SL136.tap"]
