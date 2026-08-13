#!/usr/local/bin/wish

catch { policy trusted }
package require snack

if ![info exists embed_args(conf)] {
    set embed_args(conf) 2
}

option add *background grey75
option add *font {Helvetica 12 bold}

sound s0
sound s1
sound s2

array set v {pps1 500 pps2 500 spegh 200 secth 250 sectw 500 start1 0 end1 -1 start2 0 end2 -1 bw1 100 bw2 100 winlen1 160 winlen2 160 flag 0 fzoom1 0 fzoom2 0 topfr1 8000 topfr2 8000 timeh 15}
set v(out) [audio output]
set v(in)  [audio input]

pack [frame .f1] -fill x -anchor w
frame .f1.f
pack [frame .f1.f.fc] -expand false
set v(y1) [canvas .f1.f.fc.cy -width 30 -height $v(spegh) -highlightthickness 0]
pack $v(y1) -side left -anchor n
set v(c1) [canvas .f1.f.fc.c -width 600 -height $v(spegh) -xscrollcommand [list .f1.f.fc.xscroll set] -highlightthickness 0 -closeenough 2 -cursor crosshair]
scrollbar .f1.f.fc.xscroll -orient horizontal -command [list $v(c1) xview]
pack .f1.f.fc.xscroll -side bottom -fill x
pack $v(c1) -side left
pack .f1.f -side right
$v(c1) create spectrogram 0 0 -sound s1 -height $v(spegh) -tags s1 -pixelspersecond $v(pps1) -fftlength 256 -winlength $v(winlen1)
$v(c1) create text 10 10 -text "" -tags ctext -anchor w -fill red
$v(c1) create text 10 20 -text "" -tags mtext -anchor w -fill red
$v(c1) create rectangle  -1 -1 -1 -1 -tags m -fill black -stipple gray25
snack::frequencyAxis $v(y1) 0 0 30 $v(spegh) -topfr $v(topfr1)
snack::timeAxis $v(c1) 0 [expr $v(spegh)-$v(timeh)] 600 $v(timeh) $v(pps1) -tags axis -fill red
bind $v(c1) <ButtonPress-1>   { B1Press   1 %x %y }
bind $v(c1) <Motion>          { B1Motion  1 %x %y }
bind $v(c1) <ButtonRelease-1> { B1Release 1 %x %y }
bind $v(c1) <3> { PlayMark 1 }

if {$embed_args(conf) == 1} {
    pack [ set sf [frame .f3]] -fill both -expand true
    pack [ set v(c1s) [canvas $sf.c1s -width $v(sectw) -height $v(secth) -cursor crosshair -highlightthickness 0]]
    bind $v(c1s) <Motion> { PutCrossHairs $v(c1s) %x %y }
    $v(c1s) create section 0 0 -sound s1 -width $v(sectw) -height $v(secth) -tags s1 -frame yes -fftlength 1024 -winlength 1024
    if {$tcl_platform(platform) == "windows"} {
	$v(c1s) create line -1 0 -1 $v(sectw) -tags sm1
	$v(c1s) create line -1 0 -1 $v(secth) -tags sm2
    } else {
	$v(c1s) create line -1 0 -1 $v(sectw) -tags sm1 -stipple gray50
	$v(c1s) create line -1 0 -1 $v(secth) -tags sm2 -stipple gray50
    }
    $v(c1s) create text 100 10 -tags labsect
    bind $v(c1s) <Leave> { 
	$v(c1s) coords sm1 -1 -1 -1 -1
	$v(c1s) coords sm2 -1 -1 -1 -1
    }
    bind $v(c1) <Leave> { $v(c1s) coords sm1 -1 -1 -1 -1 }
}

if {$embed_args(conf) >= 2} {
    pack [frame .f2] -fill both -expand true
    frame .f2.f
    pack [frame .f2.f.fc] -expand false
    set v(y2) [canvas .f2.f.fc.cy -width 30 -height $v(spegh) -highlightthickness 0]
    pack $v(y2) -side left -anchor n

    set v(c2) [canvas .f2.f.fc.c -width 600 -height $v(spegh) -xscrollcommand [list .f2.f.fc.xscroll set] -highlightthickness 0 -closeenough 2 -cursor crosshair]
    scrollbar .f2.f.fc.xscroll -orient horizontal -command [list $v(c2) xview]
    pack .f2.f.fc.xscroll -side bottom -fill x
    pack $v(c2) -side left
    pack .f2.f -side right
    if {$embed_args(conf) == 2} {
	$v(c2) create spectrogram 0 0 -sound s2 -height $v(spegh) -tags s2 -pixelspersecond $v(pps2) -fftlength 256 -winlength $v(winlen2)
    } else {
	$v(c2) create waveform 0 0 -sound s2 -height $v(spegh) -tags s2 -pixelspersecond $v(pps2)
    }
    if {$embed_args(conf) == 2} {
	snack::frequencyAxis $v(y2) 0 0 30 $v(spegh) -topfr $v(topfr2)
    }
    $v(c2) create text 10 10 -text "" -tags ctext -anchor w -fill red
    $v(c2) create text 10 20 -text "" -tags mtext -anchor w -fill red
    $v(c2) create rectangle  -1 -1 -1 -1 -tags m -fill black -stipple gray25
    bind $v(c2) <ButtonPress-1>   { B1Press   2 %x %y }
    bind $v(c2) <Motion>          { B1Motion  2 %x %y }
    bind $v(c2) <ButtonRelease-1> { B1Release 2 %x %y }
    bind $v(c2) <3> { PlayMark 2 }
}

pack [ frame .f] -fill x
set overview [canvas .f.a -height 110 -width 400 -highlightthickness 0]
$overview create spectrogram 0 0 -anchor nw -sound s0 -height 110 -width 400
pack $overview -side left
pack [ frame .f.rp] -side left
pack [ button .f.rp.r -text Record -fg red] -side top
bind .f.rp.r <ButtonPress-1>   {s0 record -input $v(in);.f.rp.r configure -activebackground red}
bind .f.rp.r <ButtonRelease-1> {s0 stop;.f.rp.r configure -activebackground #ececec}
pack [ button .f.rp.p -text Play -fg green4 -width 6 -command {s0 play -output $v(out)}] -side top

if {$tcl_platform(platform) != "windows"} {
    if {$tcl_platform(byteOrder) == "bigEndian"} {
	pack [ frame .f.rp.in]
	pack [ radiobutton .f.rp.in.oe -text "ext" -value Headphones -variable v(out) -width 2 -font {Helvetica 9 bold}] -side left
	pack [ radiobutton .f.rp.in.oi -text "int" -value Speaker -variable v(out) -width 2 -font {Helvetica 9 bold}] -side left
    }
    pack [ frame .f.rp.out]
    pack [ radiobutton .f.rp.out.oe -text "mic" -value Microphone -variable v(in) -width 2 -font {Helvetica 9 bold}] -side left
    pack [ radiobutton .f.rp.out.oi -text "line" -value Line -variable v(in) -width 2 -font {Helvetica 9 bold}] -side left
}
pack [ frame .f.f] -side left
pack [ frame .f.f.u -highlightthickness 0] -side top
pack [ frame .f.f.u.u] -side top 
pack [ frame .f.f.u.b] -side top
pack [ button .f.f.u.u.u -text Copy -command {Copy 1}] -side left
pack [ button .f.f.u.u.pl -text Play -command { PlayMark 1 }] -side left
pack [ button .f.f.u.u.pr -text Print -command { Print 1 }]   -side left
pack [ label .f.f.u.b.l -text "analysis bw:"] -side left
pack [ entry .f.f.u.b.e -width 4 -textvariable v(bw1)] -side left
pack [ label .f.f.u.b.l2 -text Hz] -side left
bind .f.f.u.b.e <Key-Return> { SetBW 1 }
pack [ frame .f.f2 -highlightthickness 0] -side left -fill both
pack [ frame .f.f2.u] -side top
pack [ scale .f.f2.u.s1 -variable v(pps1) -label "zoom t" -orient horizontal -from 200 -to 2000 -command {ZoomT 1} -showvalue no] -side left
pack [ scale .f.f2.u.s2 -variable v(fzoom1) -label "zoom f" -from 0 -to 6000 -orient horizontal -command {ZoomF 1} -showvalue no] -side left

if {$embed_args(conf) >= 2} {
    pack [ frame .f.f.b -highlightthickness 0] -side top
    pack [ frame .f.f.b.u] -side top
    pack [ frame .f.f.b.b] -side top
    pack [ button .f.f.b.u.d -text Copy -fg red3 -command {Copy 2}] -side left
    pack [ button .f.f.b.u.pl -text Play -fg red3 -command { PlayMark 2 }] -side left
    pack [ button .f.f.b.u.pr -text Print -fg red3 -command { Print 2 }]   -side left
    if {$embed_args(conf) == 2} {
	pack [ label .f.f.b.b.l -text "analysis bw:" -fg red3] -side left
	pack [ entry .f.f.b.b.e -width 4 -textvariable v(bw2) -fg red3] -side left
	pack [ label .f.f.b.b.l2 -text Hz -fg red3] -side left
	bind .f.f.b.b.e <Key-Return> { SetBW 2 }
    }
    pack [ frame .f.f2.b] -side bottom
    pack [ scale  .f.f2.b.s1 -variable v(pps2) -fg red3 -label "zoom t" -orient horizontal -from 200 -to 2000 -command {ZoomT 2} -showvalue no] -side left
    if {$embed_args(conf) == 2} {
	pack [ scale .f.f2.b.s2 -variable v(fzoom2) -fg red3 -label "zoom f" -from 0 -to 6000 -orient horizontal -command {ZoomF 2} -showvalue no] -side left
    }
}

proc B1Press {n x y} {
    global v embed_args

    set xc [$v(c$n) canvasx $x]
    $v(c$n) coords m [$v(c$n) canvasx $x] -1 $xc $v(spegh)
    set v(start$n) [expr int(16000 * $xc / $v(pps$n))]
    if {$embed_args(conf) == 1} {
	$v(c1s) itemconfigure s1 -start $v(start$n)
    }
    set v(flag) 1
}

proc B1Motion {n x y} {
    global v embed_args

    set xc [$v(c$n) canvasx $x]
    if {$xc < 0} { set xc 0 }
    set yc [$v(c$n) canvasy $y]
    set f [expr int($v(topfr$n) - $v(topfr$n) * $yc / $v(spegh))]
    if {$embed_args(conf) == 1} {
	PutCrossHairs $v(c1s) $f -1
    }
    if $v(flag) {
	set co [$v(c$n) coords m]
	$v(c$n) coords m [lindex $co 0] -1 $xc $v(spegh)
	$v(c$n) coords mtext $xc 20
	set v(start$n) [expr int(16000 * [lindex $co 0] / $v(pps$n))]
	set v(end$n) [expr int(16000 * $xc / $v(pps$n))]
	set t1 [format "%.3f" [expr $v(start$n) / 16000.0]]
	set t2 [format "%.3f" [expr $v(end$n) / 16000.0]]
	$v(c$n) itemconfigure mtext -text "$t1 - $t2"
	if {$embed_args(conf) == 1} {
	    $v(c1s) itemconfigure s1 -end $v(end$n)
	}
    }
    if {$f < 0.0} { set f 0.0 }
    if {$f > $v(topfr$n)} { set f [expr $v(topfr$n)] }
    set xt [expr 10 + [lindex [$v(c$n) xview] 0] * [expr int($v(pps$n) * [s$n length -units seconds])]]
    set t [format "%.3f" [expr 1.0 * $xc / $v(pps$n)]]
    if {$t < 0.0} { set t 0.0 }
    $v(c$n) coords ctext $xt 10
    if {$n == 1 || $embed_args(conf) == 2} {
	$v(c$n) itemconfigure ctext -text "frequency: $f Hz, time $t s"
    } else {
	$v(c$n) itemconfigure ctext -text "time $t s"
    }
}

proc B1Release {n x y} {
    global v embed_args

    set co [$v(c$n) coords m]
    if {[lindex $co 0] == [$v(c$n) canvasx $x]} {
	$v(c$n) coords m -1 -1 -1 -1
	$v(c$n) itemconfigure mtext -text ""
	set v(start$n) 0
	set v(end$n)   -1
    }
    set v(flag) 0
}

proc PlayMark n {
    global v

    s$n play -start $v(start$n) -end $v(end$n) -output $v(out)
}

proc Print n {
    global v

    $v(c$n) itemconfigure ctext -text ""
    set x [expr [lindex [$v(c$n) xview] 0] * [expr int($v(pps$n) * [s$n length -units seconds])]]
    snack::frequencyAxis $v(c$n) $x 0 30 $v(spegh) -topfr $v(topfr$n) -tags junk
    $v(c$n) postscript -file junk.ps -rotate true -x $x -pagewidth 26c -colormode mono
    exec lpr junk.ps
    $v(c$n) delete junk
}

proc SetBW n {
    global v

    set v(winlen$n) [expr int(16000 / $v(bw$n))]
    if {$v(winlen$n) >= 256} {
	$v(c$n) itemconfigure s$n -winlength $v(winlen$n) -fftlength 1024
    } else {
	$v(c$n) itemconfigure s$n -winlength $v(winlen$n) -fftlength 256
    }
}

proc Copy n {
    global v

    set width [expr int($v(pps$n) * [s0 length -units seconds])]
    if {$width < 890} { set width 890 }
    $v(c$n) configure -width $width -scrollregion "0 0 $width $v(spegh)"
    $v(c$n) delete axis
    snack::timeAxis $v(c$n) 0 [expr $v(spegh)-$v(timeh)] $width $v(timeh) $v(pps$n) -tags axis -fill red
    s$n copy s0
    set v(start$n) 0
    set v(end$n) [s$n length]
    $v(c$n) coords m -1 -1 -1 -1
    $v(c$n) itemconfigure mtext -text ""
}

proc PutCrossHairs {c x y} {
    global v

    if {$y == -1} {
	set xc [expr 1.0 * $x / $v(topfr1) * $v(sectw)]
    } else {
	set xc [$c canvasx $x]
    }
    set yc [$c canvasy $y]
    set f [expr int($v(topfr1) * $xc / $v(sectw))]
    if {$f < 0.0} { set f 0.0 }
    if {$f > $v(topfr1)} { set f [expr $v(topfr1)] }

    $c coords sm1 $xc 0 $xc $v(secth)
    $c coords sm2 0 $yc $v(sectw) $yc
    set db [format "%.1f" [expr -90.0 * $yc / $v(secth)]]
    $c itemconfigure labsect -text "dB: $db frequency: $f"
}

proc ZoomT {n p} {
    global v

    $v(c$n) delete axis
    set v(pps$n) $p
    $v(c$n) itemconfigure s$n -pixelspersecond $v(pps$n)
    set width [expr int($v(pps$n) * [s$n length -units seconds])]
    if {$width < 890} { set width 890 }
    $v(c$n) configure -width $width -scrollregion "0 0 $width $v(spegh)"
    snack::timeAxis $v(c$n) 0 [expr $v(spegh)-$v(timeh)] $width $v(timeh) $v(pps$n) -tags axis -fill red

    if {$v(end$n) != [s$n length]} {
	set x0 [expr int($v(start$n) * $v(pps$n) / 16000.0)]
	set x1 [expr int($v(end$n) * $v(pps$n) / 16000.0)]
	$v(c$n) coords m $x0 -1 $x1 $v(spegh)
	$v(c$n) coords mtext $x1 20
    }
}

proc ZoomF {n f} {
    global v embed_args

    set v(topfr$n) [expr 8000 - $f]
    $v(c$n) itemconfigure s$n -topfrequency $v(topfr$n)
    $v(y$n) delete all
    snack::frequencyAxis $v(y$n) 0 0 30 $v(spegh) -topfr $v(topfr$n)
    if {$embed_args(conf) == 1} {
	$v(c1s) itemconfigure s1 -top $v(topfr$n)
    }
}

update
pack propagate . false
