#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

package require snack
set db 0

option add *background grey75
option add *font {Helvetica 10 bold}

sound s
sound x
sound sin
sound nsin
sound bup

set n 968
s length $n
x length 64

sin length 16
for {set i 0} {$i < 16} {incr i} {
    sin sample $i [expr int(10000*sin(2*3.1415926535*$i/16))]
}

for {set i 0} {$i < 4} {incr i} {
    s concat sin
}
s length 2000
bup copy s

nsin copy sin
for {set i 0} {$i < 8} {incr i} {
    nsin concat nsin
}

pack [canvas .c -width 400 -height 303]
.c create waveform 200 50 -sound s -pixels 3200 -height 100 -tags p -anchor c -debug $db
.c create text 30 10 -text "Waveform" -anchor nw
.c create spectrogram 200 150 -sound s -height 100 -pixels 3200 -fftlen 512 -winlen 160 -bright 100 -cont 40 -tags [list p s] -anchor c -debug $db
.c create text 30 110 -text "Spectrogram" -anchor nw
.c create line 192 100 208 100 -arrow both -tags win -fill red
.c create text 200 110 -text "FFT window length" -fill red
.c create text 30 210 -text "Ideal spectrogram" -anchor nw
.c create waveform 200 287 -sound x -pixels 3200 -height 100 -tags p -anchor c -debug $db
snack_yAxis .c 0 100 20 100 -topfrequency 8000
snack_yAxis .c 0 200 20 100 -topfrequency 8000
.c create text 400 300 -text "t" -anchor se
.c create line 0 300 400 300 -arrow last
pack [ label .l -textvar lab -wi 50 -font {Helvetica 10 bold} -anchor w] -fill x
pack [frame .f] -fill both
pack [button .f.bp -text Play -command {s play}] -side left
set bw 160
pack [ scale .f.s1 -var bw -label "Bandwidth" -orient horiz -from 20 -to 500 -com SetBW -showvalue no] -side left
pack [ scale .f.s2 -var pps -label "Zoom" -orient horiz -from 3200 -to 20000 -com Zoom -showvalue no] -side left
set jk 0
pack [frame .f.f] -side left
pack [radiobutton .f.f.bs1 -var jk -value 0 -text Pulse -command Pulse] -anc w
pack [radiobutton .f.f.bs2 -var jk -value 1 -text Continuous -command Cont] -anc w

proc Zoom val {
    global bw pps
    .c itemconf p -pixelsp $val
    set width [expr int($pps / $bw)]
    .c coords win [expr 200 - $width/2] 100 [expr 200 + $width/2] 100
}

proc SetBW val {
    global lab pps

    set winlen [expr int(16000 / $val)]
    .c itemconf s -winlen $winlen
    set width [expr $pps * $winlen / 16000]
    .c coords win [expr 200 - $width/2] 100 [expr 200 + $width/2] 100
    set lab "Analysis bandwidth: $val Hz, FFT window: $winlen points"
}

proc Pulse {} { 
    s copy bup
    x length 64
    .c itemconf s -width 400
}

proc Cont {} { 
    s copy nsin
    x length 2000
    .c itemconf s -width 500
}
