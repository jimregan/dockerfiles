docker run -it \
  -e DISPLAY="$DISPLAY" \
  -e XDG_RUNTIME_DIR=/tmp \
  -e SDL_AUDIODRIVER=dummy \
  -e SDL_NO_MITSHM=1 -e SDL_VIDEO_X11_NO_SHM=1 \
  jimregan/joyce xjoyce

