# Fedora Core 3 in QEMU in Docker

This repo builds a Fedora Core 3 VM image and wraps it in two Docker images:

- `fc3-interactive`: boots the installed FC3 VM with noVNC and SSH for manual debugging.
- `fc3-final`: starts from the clean installed VM, installs the Tcl plugin RPM, and boots Mozilla on X.

The VM disk is created once into an intermediate Docker image named `fc3-installed`. The images above inherit that disk.

## Inputs

Put the Fedora Core 3 install ISOs on the host and point `ISO_DIR` at them. The installer script expects all four install discs (FC3 does not ship a single-DVD image):

- `disc1.iso`
- `disc2.iso`
- `disc3.iso`
- `disc4.iso`

`./build.sh` creates a merged install tree at `./tree` by running `scripts/prep-tree.sh` inside the Docker image. The prep container is privileged because it loop-mounts the ISOs. Disc 1 is copied in full (it carries `isolinux/`, `images/`, and `Fedora/base`); discs 2-4 only contribute their `Fedora/RPMS`.

## Build the Installed VM

To remove stale containers/images from earlier attempts:

```bash
./clean.sh
```

This prepares the merged install tree if needed, performs the FC3 kickstart install inside QEMU, commits the resulting disk into `fc3-installed`, then builds the interactive and final images. The installer tree defaults to `./tree`; override with `TREE_DIR=/path/to/tree` if needed.

```bash
ISO_DIR=/path/to/fc3-isos ./build.sh
```

The installer defaults to software CPU emulation with a Pentium II CPU model, since old installer kernels can hang under KVM host CPU passthrough on modern hardware. To try KVM during install:

```bash
INSTALL_USE_KVM=1 ISO_DIR=/path/to/fc3-isos ./build.sh
```

If the installer appears stuck, run it with noVNC enabled and open `http://server:6080/vnc.html`:

```bash
INSTALL_VNC=1 ISO_DIR=/path/to/fc3-isos ./build.sh
```

FC3 discs boot via isolinux, not a boot floppy, so the installer extracts `isolinux/vmlinuz` and `isolinux/initrd.img` straight from disc 1 and boots them with QEMU's `-kernel`/`-initrd`. During install, the container serves the merged tree and `ks.cfg` over HTTP with Python's built-in server. QEMU user networking exposes the container side to the guest installer as `10.0.2.2`, so the boot command uses `ks=http://10.0.2.2:8000/ks.cfg` and the kickstart uses `url --url http://10.0.2.2:8000/`. This avoids relying on FC3 anaconda to discover a separate floppy or synthetic hard disk just to read kickstart.

The installer NIC defaults to QEMU's `pcnet` model because FC3's early installer environment is more likely to carry the `pcnet32` driver. Override it if needed:

```bash
INSTALL_NET_MODEL=rtl8139 ISO_DIR=/path/to/fc3-isos ./build.sh
```

The runtime VM scripts default to `rtl8139`, which FC3 handles with the `8139too` driver. Override it only if you also reconfigure networking inside the guest:

```bash
RUN_NET_MODEL=pcnet docker compose up interactive
```

When kickstart is fetched successfully, the Docker log should show Python HTTP requests for `/ks.cfg` and then the install tree. If the console stops after early storage messages such as `No volume groups found` and there is no `GET /ks.cfg`, the installer has not brought up networking.

FC3 still uses some older kickstart commands. In particular, `langsupport --default=en_US en_US` is needed to avoid the interactive Language Support screen even though newer Fedora/RHEL kickstarts fold this into `lang`.

While the installer is running, check whether it is still doing useful work from another shell:

```bash
./scripts/install-status.sh
```

The installer container defaults to `fc3-install`. To use a different name:

```bash
INSTALL_CONTAINER=my-fc3-install ISO_DIR=fc3 ./build.sh
INSTALL_CONTAINER=my-fc3-install ./scripts/install-status.sh
```

The guest root password defaults to `rootpassword`. To change it for the automation and the kickstart, update `kickstart.cfg` and run with:

```bash
ROOT_PASSWORD=your-password ISO_DIR=/path/to/fc3-isos ./build.sh
```

## Image 1: Interactive Development

```bash
docker compose up interactive
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`

If SSH is not up in an older installed image, log in through noVNC as root and run:

```bash
/sbin/chkconfig sshd on
/sbin/service sshd start
```

The host `./rpmbuild` tree is mounted into the Docker container at `/rpmbuild`. The FC3 guest does not see Docker volumes directly; copy the mounted tree into the guest after it boots:

```bash
docker compose exec interactive bash /fc3/scripts/sync-rpmbuild-to-guest.sh
```

That copies `/rpmbuild` in the container to `/root/rpmbuild` in the guest. Use this VM for manual debugging or source inspection; the final image uses the published Tcl plugin RPM by default.

## Image 2: Final Mozilla Runtime

The final image downloads the FC3 Tcl plugin RPM release by default. Override the release URL if needed with `TCLPLUGIN_RPM_URL`.

```bash
docker compose --profile final build final
docker compose --profile final up final
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`

During the final image build, the RPM is installed into the guest disk and `/root/.xinitrc` is configured to launch Mozilla directly under X, without a desktop session. The final container serves local `lab/` files at `http://www.speech.kth.se/labs/analysis/` for the guest, with `.tcl` served as `application/x-tcl`. On boot, the VM starts X on QEMU's VNC display and opens that URL.

## Tcl Plugin Notes

FC3 ships Mozilla 1.7 and Tcl/Tk 8.4, which is what the last version of the Tcl plugin needs (the plugin's minimum is Mozilla 1.0). The target plugin artifact should be installed by the RPM into Mozilla's plugin directory, usually:

```text
/usr/lib/mozilla/plugins/
```

The historical Tcl plugin build usually needs Tcl/Tk headers, X11 headers, and NPAPI headers. The kickstart installs Tcl/Tk, X.org (`xorg-x11`, replacing XFree86 as of FC3), Mozilla, compiler, and rpm-build packages; if Mozilla's RPM does not include the NPAPI headers, put the matching Mozilla 1.7 headers in `rpmbuild/SOURCES` and reference them from the spec.
