# CDC 1700 SIMH Docker Image

This builds a small Debian-based image containing the Open SIMH `cdc1700`
emulator and the upstream CDC 1700 MSOS helper scripts:

- `msosInstall.simh`
- `msosRun.simh`
- `CDC1700-MSOS.txt`

The MSOS distribution tape is downloaded from bitsavers at build time with
`wget` and baked into the image. Put runtime files in a mounted `data/`
directory so generated disk images and printer logs persist outside the
container.

## Build

```sh
docker build -t cdc1700 .
```

To build a specific Open SIMH branch, tag, commit-ish, or tape URL:

```sh
docker build --build-arg SIMH_REF=master -t cdc1700 .
docker build --build-arg MSOS_TAPE_URL=https://bitsavers.org/bits/CDC/1700_Cyber18/20100524/MSOS5_SL136.tap -t cdc1700 .
```

## Install MSOS 5

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 install
```

This runs the upstream `msosInstall.simh` script. It expects
the baked-in `MSOS5_SL136.tap` and creates:

- `MSOS5-A.dsk`
- `MSOS5-B.dsk`
- `MSOSinstall.lpt`

To install from a custom tape mounted in `data/`:

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 install install.tap
```

## Run The Installed System

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 run
```

This runs the upstream `msosRun.simh` script against `MSOS5-A.dsk` and
`MSOS5-B.dsk`. Line-printer output is written to a dated `.lpt` file in
`data/`.

## Raw SIMH Access

Run the emulator directly:

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 simh
```

Run an arbitrary SIMH command file:

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 simh my-script.simh
```

Open a shell in the image:

```sh
docker run --rm -it -v "$PWD/data:/data" cdc1700 shell
```
