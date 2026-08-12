#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

package require snack

sound s

option add *background grey75
option add *font {Helvetica 12 bold}

set out [audio output]
set in  [audio input]

pack [frame .f] -fill both -expand true
pack [set c [canvas .f.c -width 400 -height 200]]
$c create rectangle 0 0 400 30    -fill grey60 -stipple gray50
$c create rectangle 0 170 400 200 -fill grey60 -stipple gray50
$c create line 0 30 400 30   -fill red -width 2
$c create line 0 170 400 170 -fill red -width 2
$c create waveform 0 0 -sound s -width 400 -pixelspersecond 300 -height 200 -limit 32767

pack [frame .fb -height 40] -fill x
place [button .fb.r -text {Push down to record}] -relx .25 -rely .5 -anchor c
place [button .fb.p -text Play -command {s play -output $out} -width 18] -relx .75 -rely .5 -anchor c
bind .fb.r <ButtonPress-1>   {s record -input $in; .fb.r configure -activebackground red}
bind .fb.r <ButtonRelease-1> {s stop;   .fb.r configure -activebackground #ececec}

set playgain [playGain]
set recgain  [recGain]
pack [frame .fg]
if {$tcl_platform(platform) == "unix"} {
    pack [scale .fg.sr -label {Record gain} -variable recgain -orient horizontal -command recGain -length 200] -side left
}
pack [scale .fg.sp -label {Play volume} -variable playgain -orient horizontal -command playGain -length 200] -side left

if {$tcl_platform(platform) == "unix"} {
    pack [ frame .fp] -fill x
    if {$tcl_platform(byteOrder) == "bigEndian"} {
	foreach jack [audio outputs] {
	    pack [ radiobutton .fp.r$jack -text $jack -value $jack -variable out -command "audio output $jack" -font {Helvetica 11 bold}] -side left
	}
    }
    foreach jack [audio inputs] {
	pack [ radiobutton .fp.r$jack -text $jack -value $jack -variable in -command "audio input $jack" -font {Helvetica 11 bold}] -side left
    }
}


