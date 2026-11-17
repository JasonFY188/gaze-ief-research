"""
capture_server.py - https page for the iPhone: turn on the spot and take a photo at exact angles
(every 30/45/60 deg, measured by the phone's gyroscope), e.g. as input for VGGT 3D reconstruction.

Usage:
  python turn_capture/capture_server.py --port 8000
Open the printed https:// address on the iPhone (same Wi-Fi). The certificate is self-made, so Safari warns:
tap "Show Details" -> "visit this website" -> "Visit Website". iPhones only give the motion sensor and the
camera to https pages, which is why this is not plain http.

Photos:  data\\captures\\<session>\\spot1_tiltp0_yaw000.jpg ...  and  poses.jsonl (measured yaw / tilt / roll per photo).
Certificate: turn_capture/certs/cert.pem + key.pem (bash turn_capture/make_certs.sh <PC IP>).
"""
import argparse
import json
import re
import socket
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data" / "captures"           # not committed (data/ is in .gitignore)
CERTS = HERE / "certs"                       # made by make_certs.sh, never committed
SAFE = re.compile(r"[^A-Za-z0-9_-]+")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        print(f"  page request from {self.client_address[0]}: {self.path}", flush=True)
        if urlparse(self.path).path in ("/", "/index.html"):
            return self.send(200, (HERE / "capture.html").read_bytes(), "text/html; charset=utf-8")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/shot":
            return self.send(404, b"not found", "text/plain")
        q = parse_qs(u.query)
        session = SAFE.sub("_", q.get("session", ["unnamed"])[0])[:80] or "unnamed"
        name = SAFE.sub("_", q.get("name", ["shot"])[0])[:80] or "shot"
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 40_000_000:
            return self.send(400, b"no image", "text/plain")
        data = self.rfile.read(n)
        if data[:2] != b"\xff\xd8":
            return self.send(400, b"not a jpeg", "text/plain")
        try:
            meta = json.loads(q.get("meta", ["{}"])[0])
        except json.JSONDecodeError:
            meta = {}
        d = OUT / session
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.jpg").write_bytes(data)
        with open(d / "poses.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(meta, file=f"{name}.jpg"), ensure_ascii=False) + "\n")
        print(f"  {session}/{name}.jpg  yaw {meta.get('yaw')}  tilt {meta.get('tilt')}  roll {meta.get('roll')}  "
              f"{meta.get('w')}x{meta.get('h')}", flush=True)
        self.send(200, b'{"ok":true}', "application/json")


class TLSServer(ThreadingHTTPServer):
    def get_request(self):
        try:
            return super().get_request()
        except (ssl.SSLError, OSError) as e:
            print(f"  connection failed during https setup: {e}", flush=True)
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cert, key = CERTS / "cert.pem", CERTS / "key.pem"
    if not cert.exists():
        sys.exit(f"missing {cert} - run: bash turn_capture/make_certs.sh <this PC IP>  (see docs/ACTIVE_CAMERA.md)")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = TLSServer((args.host, args.port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "<this PC's IP>"
    print(f"photos go to {OUT}")
    print(f"open on the iPhone:  https://{ip}:{args.port}   (Safari warns once: Show Details -> visit this website)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
