#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

package require snack
option add *background grey75

sound s

pack [ canvas .c -height 500 -width 500 -highlightthickness 0]
.c create section 0 0 -sound s -height 500 -width 500 -fftlen 512 -winlen 512 -frame 1
pack [frame .f -height 35] -fill x
place [ button .f.br -text Record] -relx .4 -anchor n
place [ button .f.bp -text Play -wi 6 -command {s play}] -relx .6 -anchor n
bind .f.br <ButtonPress-1>   {s record}
bind .f.br <ButtonRelease-1> {s stop}
