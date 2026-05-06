# RHL 7.2 in QEMU in Docker

This repo builds a Red Hat Linux 7.2 VM image and wraps it in three Docker images:

- `rhl72-interactive`: boots the installed RHL 7.2 VM with noVNC and SSH so you can debug the old Tcl plugin build manually.
- `rhl72-builder`: boots a clean copy of that VM, copies in `./rpmbuild`, runs `rpmbuild`, and writes RPMs to `./output`.
- `rhl72-final`: starts from the clean installed VM, installs the RPMs from `./output`, and boots Mozilla with the Tcl plugin installed.

The RHL VM disk is created once into an intermediate Docker image named `rhl72-installed`. The three images above inherit that disk.

## Inputs

Put the Red Hat Linux 7.2 install ISOs on the host and point `ISO_DIR` at them. The installer script expects:

- `disc1.iso`
- `disc2.iso`

The old Tcl plugin RPM inputs should use the normal rpmbuild layout under `./rpmbuild`:

```text
rpmbuild/
  SOURCES/
  SPECS/
```

At minimum, `rpmbuild/SPECS` must contain one `.spec` file before running the builder.

## Build the Installed VM

This performs the RHL 7.2 kickstart install inside QEMU, commits the resulting disk into `rhl72-installed`, then builds the interactive and builder images.

```bash
ISO_DIR=/path/to/rhl72-isos ./build.sh
```

The installer defaults to software CPU emulation with a Pentium III CPU model because the RHL 7.2 installer kernel can hang very early with KVM host CPU passthrough on modern Linux. To try KVM during install:

```bash
INSTALL_USE_KVM=1 ISO_DIR=/path/to/rhl72-isos ./build.sh
```

If the installer appears stuck, run it with noVNC enabled and open `http://server:6080/vnc.html`:

```bash
INSTALL_VNC=1 ISO_DIR=/path/to/rhl72-isos ./build.sh
```

The installer defaults to a generated mini boot ISO that uses the extracted RHL installer kernel/initrd and an `isolinux.cfg` containing the kickstart arguments. To boot from disc 1's original ISO boot path for manual debugging:

```bash
INSTALL_BOOT=iso INSTALL_VNC=1 ISO_DIR=/path/to/rhl72-isos ./build.sh
```

The guest root password defaults to `rootpassword`. To change it for the automation and the kickstart, update `kickstart.cfg` and run with:

```bash
ROOT_PASSWORD=your-password ISO_DIR=/path/to/rhl72-isos ./build.sh
```

## Image 1: Interactive Development

```bash
docker compose up interactive
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`

Use this VM to work out the source tarball, build dependencies, spec file, and `rpmbuild` invocation. Put the resulting files in the host `./rpmbuild` tree so the automated builder can use them.

## Image 2: RPM Builder

```bash
docker compose --profile build build builder
docker compose --profile build run --rm builder
```

The builder mounts:

- `./rpmbuild` read-only at `/rpmbuild`
- `./output` at `/output`

On success, RPMs are copied to `./output`.

## Image 3: Final Mozilla Runtime

Build this only after the builder has produced RPMs in `./output`. If `./output` contains no RPMs, the final image build stops with an explicit error.

```bash
docker compose --profile final build final
docker compose --profile final up final
```

Access:

- noVNC: `http://localhost:6080/vnc.html`
- SSH: `ssh -p 2222 root@localhost`

During the final image build, the RPMs are installed into the guest disk and `/root/.xinitrc` is configured to launch Mozilla. On boot, the VM starts X on QEMU's VNC display.

## Tcl Plugin Notes

RHL 7.2 ships Mozilla 0.9.4, which uses the NPAPI plugin interface. The target plugin artifact should be installed by the RPM into Mozilla's plugin directory, usually:

```text
/usr/lib/mozilla/plugins/
```

The historical Tcl plugin build usually needs Tcl/Tk headers, X11 headers, and NPAPI headers. The kickstart installs the RHL-side Tcl/Tk, XFree86, Mozilla, compiler, and rpm-build packages; if Mozilla's RPM does not include the NPAPI headers, put the matching Mozilla 0.9.4 headers in `rpmbuild/SOURCES` and reference them from the spec.
