#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

package require snack
option add *background grey75

sound s

pack [ spectrogram .s -sound s -height 200]
pack [frame .f -height 35] -fill x
place [ button .f.br -text Record] -relx .4 -anchor n
place [ button .f.bp -text Play -width 6 -command {s play}] -relx .6 -anchor n
bind .f.br <ButtonPress-1>   {s record}
bind .f.br <ButtonRelease-1> {s stop}
