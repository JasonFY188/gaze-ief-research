#!/usr/bin/env bash
# make_certs.sh - HTTPS certificate for the Turn Capture page (iPhones only give the camera and the
# motion sensor to https pages).
#
#   bash turn_capture/make_certs.sh 192.168.1.23 [more IPs of this PC, e.g. a phone-hotspot address]
#
# Makes, in turn_capture/certs/ (never committed):
#   ca.pem / ca.key          a small local CA that may only sign PRIVATE addresses (name constraints)
#   cert.pem / key.pem       the server certificate for the given IPs (serverAuth, IP SANs)
#   TurnCapture_CA.cer       the CA in DER form: send it to the iPhone, install the profile, then
#                            Settings > General > About > Certificate Trust Settings > turn it on.
# Remove the profile from the phone when you no longer need it.
# On Windows run it in Git Bash (uses Git's openssl).
set -euo pipefail
export MSYS_NO_PATHCONV=1            # Git Bash: keep "/CN=..." as is
[ $# -ge 1 ] || { echo "usage: bash $0 <PC IP> [more IPs]"; exit 1; }
D="$(cd "$(dirname "$0")" && pwd)/certs"
mkdir -p "$D"; cd "$D"

if [ ! -f ca.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 -sha256 -keyout ca.key -out ca.pem \
    -subj "/CN=Turn Capture local CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -addext "nameConstraints=critical,permitted;IP:10.0.0.0/255.0.0.0,permitted;IP:172.16.0.0/255.240.0.0,permitted;IP:192.168.0.0/255.255.0.0"
fi

SAN=""
for ip in "$@"; do SAN="${SAN}IP:${ip},"; done
SAN="${SAN%,}"
openssl req -newkey rsa:2048 -nodes -keyout key.pem -out server.csr -subj "/CN=turn-capture"
printf "basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\n" "$SAN" > server.ext
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -days 825 -sha256 -extfile server.ext -out cert.pem
openssl x509 -in ca.pem -outform der -out TurnCapture_CA.cer
echo "done: $D (server certificate for $SAN)"
