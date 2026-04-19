#!/bin/sh
set -eu

scripts_dir=/opt/cdc1700
default_tape=${scripts_dir}/MSOS5_SL136.tap

usage() {
    cat <<EOF
CDC 1700 SIMH container

Usage:
  docker run --rm -it -v "\$PWD/data:/data" cdc1700 help
  docker run --rm -it -v "\$PWD/data:/data" cdc1700 install [tape.tap]
  docker run --rm -it -v "\$PWD/data:/data" cdc1700 run
  docker run --rm -it -v "\$PWD/data:/data" cdc1700 simh [args...]
  docker run --rm -it -v "\$PWD/data:/data" cdc1700 shell

The install command creates MSOS5-A.dsk, MSOS5-B.dsk, and MSOSinstall.lpt
in /data. The run command uses those disk images and writes dated .lpt output.

The default MSOS tape is baked into the image at ${default_tape}.
MSOS installation notes are in ${scripts_dir}/CDC1700-MSOS.txt.
EOF
}

cmd=${1:-help}
case "$cmd" in
    help|-h|--help)
        usage
        ;;
    install)
        shift
        tape=${1:-$default_tape}
        if [ ! -f "$tape" ]; then
            cat >&2 <<EOF
Missing $tape in /data.

Use the default baked-in tape by omitting the tape argument, or place a custom
tape in the mounted data directory and pass its path/name to install.
EOF
            exit 1
        fi
        exec cdc1700 "${scripts_dir}/msosInstall.simh" "$tape"
        ;;
    run)
        if [ ! -f MSOS5-A.dsk ] || [ ! -f MSOS5-B.dsk ]; then
            cat >&2 <<EOF
Missing MSOS5-A.dsk or MSOS5-B.dsk in /data.

Run the install command first.
EOF
            exit 1
        fi
        exec cdc1700 "${scripts_dir}/msosRun.simh"
        ;;
    simh)
        shift
        exec cdc1700 "$@"
        ;;
    shell|sh)
        exec /bin/sh
        ;;
    *)
        exec cdc1700 "$@"
        ;;
esac
