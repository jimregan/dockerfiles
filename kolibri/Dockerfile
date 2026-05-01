FROM debian:bullseye AS builder

RUN apt-get update && apt-get install -y wget p7zip-full && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/latest-iso.7z https://builds.kolibrios.org/en_US/latest-iso.7z \
    && 7z e /tmp/latest-iso.7z -o/iso/ "*.iso" \
    && rm /tmp/latest-iso.7z

FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    qemu-system-x86 \
    novnc \
    python3-websockify \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /iso/*.iso /kolibri.iso

EXPOSE 5900 6080

CMD qemu-system-i386 -cdrom /kolibri.iso -boot d -m 64 -vnc :0 -k en-us & \
    exec websockify --web=/usr/share/novnc 6080 localhost:5900
