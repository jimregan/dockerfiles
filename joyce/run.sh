REAL_DISPLAY=$MAC:0

docker run --rm -it --net=host \
  -e DISPLAY="$REAL_DISPLAY" \
  -e XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  -v "${XAUTHORITY:-$HOME/.Xauthority}:${XAUTHORITY:-$HOME/.Xauthority}:ro" \
  -e XDG_RUNTIME_DIR=/tmp \
  -e SDL_AUDIODRIVER=dummy \
  -e SDL_NO_MITSHM=1 -e SDL_VIDEO_X11_NO_SHM=1 \
  jimregan/joyce xjoyce

