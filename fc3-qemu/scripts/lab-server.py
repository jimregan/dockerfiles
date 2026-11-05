#!/usr/bin/env python3
# Serves /www on port 80 so the FC3 guest can reach
# http://www.speech.kth.se/labs/analysis/ (which it resolves to this
# container via the QEMU usermode gateway, 10.0.2.2).
import http.server


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".tcl": "application/x-tcl",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/www", **kwargs)


if __name__ == "__main__":
    http.server.HTTPServer(("0.0.0.0", 80), Handler).serve_forever()
