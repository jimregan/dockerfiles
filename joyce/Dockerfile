# ---- build stage ----
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential automake pkg-config wget ca-certificates \
    libxml2-dev libpng-dev libsdl1.2-dev \
 && rm -rf /var/lib/apt/lists/*

# Build JOYCE
WORKDIR /tmp
RUN wget https://www.seasip.info/Unix/Joyce/joyce-2.5.4.tar.gz
RUN tar zxvf joyce-2.5.4.tar.gz
WORKDIR /tmp/joyce-2.5.4
RUN ./configure
RUN make -j"$(nproc)" && make install

# ---- runtime stage ----
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Runtime libs only (SDL 1.2 compat + XML + PNG + X11 client libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl1.2-compat libxml2 libpng16-16 \
    libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 libxinerama1 \
    # optional: for quick testing, include x11-apps (xclock, xeyes)
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# Bring over installed JOYCE binaries from the build stage
COPY --from=build /usr/local/ /usr/local/

# Default entrypoint can be JOYCE; override at `docker run` if needed

