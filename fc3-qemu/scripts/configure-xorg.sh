#!/bin/bash
# Configure Xorg for QEMU cirrus video and PS/2 mouse.
set -euo pipefail

cat > /etc/X11/xorg.conf <<'EOF'
Section "ServerLayout"
    Identifier "Default Layout"
    Screen 0 "Screen0" 0 0
    InputDevice "Mouse0" "CorePointer"
    InputDevice "Keyboard0" "CoreKeyboard"
EndSection

Section "InputDevice"
    Identifier "Keyboard0"
    Driver "kbd"
    Option "XkbModel" "pc105"
    Option "XkbLayout" "us"
EndSection

Section "InputDevice"
    Identifier "Mouse0"
    Driver "mouse"
    Option "Protocol" "IMPS/2"
    Option "Device" "/dev/input/mice"
    Option "ZAxisMapping" "4 5"
EndSection

Section "Device"
    Identifier "Videocard0"
    Driver "cirrus"
EndSection

Section "Monitor"
    Identifier "Monitor0"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection

Section "Screen"
    Identifier "Screen0"
    Device "Videocard0"
    Monitor "Monitor0"
    DefaultDepth 16
    SubSection "Display"
        Depth 16
        Modes "1024x768" "800x600"
    EndSubSection
EndSection
EOF
