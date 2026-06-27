#!/usr/bin/env bash
# Generate a self-signed TLS cert for Hallo4 Studio so browsers grant a "secure
# context" — required for getUserMedia (webcam/mic) when the studio is opened
# from another computer over the LAN (plain http://<lan-ip> blocks camera access).
#
# Usage:
#   bash scripts/make_studio_cert.sh                 # auto-detects LAN IP
#   bash scripts/make_studio_cert.sh 192.168.1.64    # pin an IP/hostname
#
# Then run the backend over HTTPS:
#   uvicorn studio.backend.hallo4_studio.app:app --host 0.0.0.0 --port 8443 \
#       --ssl-keyfile studio_data/certs/studio.key --ssl-certfile studio_data/certs/studio.crt
#
# Open https://<lan-ip>:8443/ and accept the self-signed cert once per browser.
set -euo pipefail

CERT_DIR="${HALLO4_STUDIO_CERT_DIR:-studio_data/certs}"
mkdir -p "$CERT_DIR"

IP="${1:-}"
if [[ -z "$IP" ]]; then
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
IP="${IP:-127.0.0.1}"
echo "Issuing self-signed cert for: $IP (and localhost / 127.0.0.1)"

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$CERT_DIR/studio.key" \
  -out "$CERT_DIR/studio.crt" \
  -days 825 \
  -subj "/CN=$IP" \
  -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost"

chmod 600 "$CERT_DIR/studio.key"
echo "Wrote $CERT_DIR/studio.crt and $CERT_DIR/studio.key"
