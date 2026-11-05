#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

package require snack
option add *background grey75

sound s

set c [canvas .s -height 200 -width 400 -highlightthickness 0]
$c create spectrogram 0 0 -anchor nw -sound s -height 200 -width 400
pack $c
pack [frame .f -height 35] -fill x
place [ button .f.br -text Record] -relx .4 -anchor n
place [ button .f.bp -text Play -width 6 -command {s play}] -relx .6 -anchor n
bind .f.br <ButtonPress-1>   {s record}
bind .f.br <ButtonRelease-1> {s stop}
