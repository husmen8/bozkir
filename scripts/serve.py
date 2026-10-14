"""Serve the viewer locally, with caching off.

    python scripts/serve.py            # http://localhost:8000
    python scripts/serve.py --port 8080

`python -m http.server` sends no cache headers, so a browser may keep an
old viewer.js or module after an edit and run it under a new page. This
sends `Cache-Control: no-store` for every file.
"""

import argparse
import functools
import http.server
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


class NoStore(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    handler = functools.partial(NoStore, directory=str(WEB))
    with http.server.ThreadingHTTPServer(("", args.port), handler) as httpd:
        print(f"  serving {WEB} at http://localhost:{args.port} (no caching); Ctrl+C stops")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
