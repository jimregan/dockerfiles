"""Fetch the Gutenberg HTML source texts named in the inventory.

    uv run python -m librivox_mult.fetch_gutenberg [--id N ...] [--force]

Writes ``<out>/gutenberg/pg<N>-images.html`` plus ``<out>/gutenberg/manifest.jsonl``
(id, file, url, sha256, bytes, fetched/copied date, source). Gutenberg
re-releases files, so the sha256 in the manifest, not the URL, identifies the
text a run used.

Texts already on hand locally (``--local DIR``, files named
``pg<N>-images.html``) are copied, not downloaded. Existing outputs are kept
unless --force. Requests are sequential with a delay (Gutenberg asks for
no bulk crawling of the website; this fetches a few dozen books once).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import sys
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("gutenberg")
URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}-images.html"
UA = "librivox-mult corpus build (research use; one-off fetch of ~60 books)"
LOCAL = [Path("/src-projects/librivox-mult-work-files/text")]


def inventory_ids(inv: Path) -> list[int]:
    ids = set()
    for line in inv.read_text().splitlines():
        ids.update(json.loads(line)["gutenberg_ids"])
    return sorted(ids)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("/out/inventory.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--id", type=int, action="append", help="only these ids")
    ap.add_argument("--local", type=Path, action="append", default=None,
                    help="dirs searched recursively for pg<N>-images.html to copy")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(a.out / "logs" / "gutenberg.log"), logging.StreamHandler(sys.stderr)],
    )
    gdir = a.out / "gutenberg"
    gdir.mkdir(parents=True, exist_ok=True)
    man_path = gdir / "manifest.jsonl"
    manifest = {}
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            r = json.loads(line)
            manifest[r["id"]] = r
    ids = a.id or inventory_ids(a.inventory)
    local = {}
    for d in a.local or LOCAL:
        for p in d.rglob("pg*-images.html"):
            local[int(p.name[2:].split("-")[0])] = p
    today = dt.date.today().isoformat()
    fetched = 0
    for n in ids:
        dst = gdir / f"pg{n}-images.html"
        if dst.exists() and n in manifest and not a.force:
            continue
        if n in local:
            shutil.copyfile(local[n], dst)
            entry = {"source": "local-copy", "url": str(local[n]), "date": today}
            log.info("pg%d: copied from %s", n, local[n])
        else:
            if fetched:
                time.sleep(a.delay)
            url = URL.format(n=n)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception as e:  # noqa: BLE001 - recorded; others continue
                log.error("pg%d: %s: %s", n, url, e)
                continue
            fetched += 1
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dst)
            entry = {"source": "download", "url": url, "date": today}
            log.info("pg%d: %d bytes", n, len(data))
        manifest[n] = {"id": n, "file": dst.name, "sha256": sha256(dst),
                       "bytes": dst.stat().st_size, **entry}
        with man_path.with_suffix(".tmp").open("w") as f:
            for k in sorted(manifest):
                f.write(json.dumps(manifest[k], sort_keys=True) + "\n")
        man_path.with_suffix(".tmp").replace(man_path)
    missing = [n for n in ids if n not in manifest]
    log.info("%d ids, %d in manifest, missing: %s", len(ids), len(manifest), missing)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
