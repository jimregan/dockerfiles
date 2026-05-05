# RHL 7.2 in QEMU in Docker

Historical recreation: Red Hat Linux 7.2 running Apache 1.3 with mod_tcl.

## Prerequisites

- The three RHL 7.2 ISOs placed in `./isos/` as `disc1.iso`, `disc2.iso`, `disc3.iso`
- Docker with `--privileged` support and `/dev/kvm` available on the host

## First-time setup

### 1. Build the base image

```bash
docker build -f Dockerfile.base -t rhl72-base .
```

### 2. Prepare the install tree and extract the boot kernel

Run these on the **host** (they need loop mount access):

```bash
# Combine the three ISOs into a single package tree
sudo ./scripts/prep-tree.sh isos/disc1.iso isos/disc2.iso isos/disc3.iso ./tree

# Pull out the installer kernel/initrd from disc1
sudo ./scripts/extract-boot.sh isos/disc1.iso ./isos

# Serve the tree and kickstart file over HTTP on port 8080
# (QEMU guest will reach this as 10.0.2.2:8080)
python3 -m http.server 8080 --directory .
```

### 3. Install the VM

```bash
# This writes disk/rhl72.qcow2 (persisted in the Docker volume)
docker compose run --rm interactive /rhl72/scripts/install-vm.sh
```

## Interactive development (Image 1)

```bash
docker compose up interactive
# Connect VNC client to localhost:5900
# Or SSH: ssh -p 2222 root@localhost   (password: rootpassword)
```

Use this session to:
- Find the mod_tcl source version that works with Apache 1.3 + Tcl 8.3
- Work out the `rpmbuild` invocation
- Write the `.spec` file into `rpmbuild/SPECS/`

## Automated build (Image 2)

Once the spec file is in `rpmbuild/SPECS/`:

```bash
# Copy the developed disk image out of the volume first
docker run --rm -v rhl72-disk:/disk -v $(pwd)/disk:/out \
    alpine cp /disk/rhl72.qcow2 /out/rhl72.qcow2

# Build the automated image and run it
docker build -f Dockerfile.build -t rhl72-build .
docker compose --profile build run --rm build
# Output RPM will be in ./output/
```
