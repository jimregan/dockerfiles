# Fedora Core 3 in QEMU in Docker

This repo builds a Fedora Core 3 VM image and wraps it in two Docker images:

- `fc3-interactive`: the installed FC3 VM with noVNC and SSH for manual debugging, plugins and all.
- `fc3-final`: the same VM, without the SSH/debugging-oriented extras.

The Tcl plugin and Snack Sound Toolkit RPMs are installed once, during the kickstart install itself, so both images inherit them along with X-boots-to-Mozilla behavior straight from `fc3-installed` — `fc3-final` is really just `fc3-interactive` without the workspace/rpmbuild mounts. Both still separately copy in and serve `lab/` for the guest to browse.

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

`kickstart.cfg`'s `%post` also fetches and installs the Tcl plugin and Snack Sound Toolkit RPMs directly, right after the base packages — the installer container downloads them (it has normal outbound internet access; the guest itself doesn't need to reach past the QEMU usermode gateway) and serves them from the same local install-tree HTTP server the kickstart itself uses, so the guest only ever fetches from `10.0.2.2`. `TCLPLUGIN_RPM_URL` and `SNACK_RPM_URL` default to release assets published alongside each other; override either if needed:

```bash
SNACK_RPM_URL=https://example.com/my-snack.rpm ISO_DIR=/path/to/fc3-isos ./build.sh
```

## Image 1: Interactive Development

```bash
docker compose up interactive
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`
- Audio stream: `http://localhost:8000/fc3.mp3` (also playable from the "Audio" button on the noVNC page)

If SSH is not up in an older installed image, log in through noVNC as root and run:

```bash
/sbin/chkconfig sshd on
/sbin/service sshd start
```

The host `./rpmbuild` tree is mounted into the Docker container at `/rpmbuild`. The FC3 guest does not see Docker volumes directly; copy the mounted tree into the guest after it boots:

```bash
docker compose exec interactive bash /fc3/scripts/sync-rpmbuild-to-guest.sh
```

That copies `/rpmbuild` in the container to `/root/rpmbuild` in the guest. Use this VM when you also want SSH/console access alongside the working lab demos; `final` is the same guest disk without the workspace/rpmbuild mounts.

## Image 2: Final Mozilla Runtime

```bash
docker compose --profile final build final
docker compose --profile final up final
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`
- Audio stream: `http://localhost:8000/fc3.mp3` (also playable from the "Audio" button on the noVNC page)

## Boot, Lab Files, and Audio

The shared kickstart (`kickstart.cfg`) bakes both the Tcl plugin/Snack RPM install and X-on-boot into `fc3-installed` itself, so both images boot straight to a full GNOME session (via `/root/.xinitrc`, started from `rc.local`) with Mozilla already open at `http://www.speech.kth.se/labs/analysis/` — no manual `startx` needed. `.xinitrc` launches Mozilla in the background and then `exec`s `gnome-session`, so the guest gets a real window manager and panel over VNC instead of an undecorated, un-movable Mozilla window with no way to reach its own address bar. `/etc/hosts` on the guest resolves `www.speech.kth.se` to `10.0.2.2`, the QEMU usermode gateway, which SLIRP forwards straight through to the container.

Each container runs `scripts/lab-server.py`, a small `http.server` bound to port 80 that serves `./lab/` (copied into the image at `/www/labs/analysis/`) with `.tcl` mapped to `application/x-tcl`. Nothing above that path is served — there's no wider site to fake, just the one directory the demos expect.

The `lab/` demos are KTH's own Snack-based speech tools and drive real audio hardware (`package require snack`, `audio output`, `s play`), so there's no VNC audio channel to tap. Instead:

- QEMU emulates an ES1370 (Ensoniq AudioPCI) sound card, and its `wav` audiodev writes PCM to a FIFO (`scripts/common.sh`'s `qemu_sound_device_args`/`start_audio_stream`) instead of a static file.
- `ffmpeg` reads that FIFO live and re-encodes it to MP3, pushed to an Icecast server (`icecast2`, both installed in `Dockerfile.base`) at `/fc3.mp3` on port 8000.
- noVNC's page has a small vendored script (`scripts/novnc/fc3-audio-button.js`, injected into `vnc.html` at image build time) adding an "Audio" toggle button that plays that Icecast stream directly — no third-party CDN dependency, just the stream itself.

## Tcl Plugin and Snack Notes

FC3 ships Mozilla 1.7 and Tcl/Tk 8.4, which is what the last version of the Tcl plugin needs (the plugin's minimum is Mozilla 1.0). The target plugin artifact should be installed by the RPM into Mozilla's plugin directory, usually:

```text
/usr/lib/mozilla/plugins/
```

The historical Tcl plugin build usually needs Tcl/Tk headers, X11 headers, and NPAPI headers. The kickstart installs Tcl/Tk, X.org (`xorg-x11`, replacing XFree86 as of FC3), Mozilla, compiler, and rpm-build packages; if Mozilla's RPM does not include the NPAPI headers, put the matching Mozilla 1.7 headers in `rpmbuild/SOURCES` and reference them from the spec.

The `lab/` demos also need the Snack Sound Toolkit (`package require snack`), which is a separate Tcl extension from the Tcl plugin itself — it's what actually plays and visualizes audio, while the Tcl plugin only lets Mozilla embed and run `.tcl` files at all. Both `tclplugin-3.1-1.i386.rpm` and `snack-2.2.10-1.i386.rpm` are installed the same way, one `rpm -Uvh` each (not combined into one transaction — FC3's rpm has thrown spurious dependency failures against unrelated already-installed packages when they're combined) from `kickstart.cfg`'s `%post`, fetched from `TCLPLUGIN_RPM_URL`/`SNACK_RPM_URL` by `scripts/install-vm.sh` before the guest ever boots.

The plugin loading is not enough on its own for `package require snack` to succeed inside it: `snack.rpm` installs its Tcl package under `/usr/lib/snack2.2/`, but the plugin's own `plugmain.tcl` only adds its own plugin directory to `auto_path`, not `/usr/lib`. Ordinarily Tcl finds `/usr/lib/snack2.2` on its own by scanning one level below `tcl_library`'s parent, but that resolution normally goes through "relative to the running executable", which breaks when Tcl is embedded as a `.so` dlopen'd into Mozilla rather than run as its own binary. `/root/.xinitrc` works around this by exporting `TCLLIBPATH=/usr/lib/snack2.2` before launching Mozilla — `TCLLIBPATH` entries are appended to `auto_path` unconditionally during Tcl's startup, independent of how `tcl_library` resolved, and are inherited by the plugin's interpreter since it runs in-process.
