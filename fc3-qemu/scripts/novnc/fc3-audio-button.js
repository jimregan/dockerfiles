// Toggle button for the FC3 guest's audio, streamed out via Icecast
// (see scripts/common.sh's start_audio_stream). No external dependencies —
// noVNC's page has no other way to carry sound from the emulated guest.
(function () {
    "use strict";

    var streamUrl = (document.currentScript && document.currentScript.dataset.streamPath) ||
        (window.location.protocol + "//" + window.location.hostname + ":8000/fc3.mp3");

    var audio = new Audio(streamUrl);
    audio.preload = "none";

    var btn = document.createElement("button");
    btn.textContent = "Audio: off";
    btn.title = "Toggle FC3 guest audio";
    btn.style.cssText = [
        "position:fixed", "top:8px", "right:8px", "z-index:10000",
        "padding:6px 10px", "font-size:13px", "cursor:pointer",
        "background:#333", "color:#fff", "border:1px solid #555",
        "border-radius:4px"
    ].join(";");

    var playing = false;
    btn.addEventListener("click", function () {
        if (playing) {
            audio.pause();
            btn.textContent = "Audio: off";
            playing = false;
        } else {
            audio.play().catch(function (err) {
                console.error("fc3-audio-button: playback failed", err);
            });
            btn.textContent = "Audio: on";
            playing = true;
        }
    });

    window.addEventListener("DOMContentLoaded", function () {
        document.body.appendChild(btn);
    });
})();
