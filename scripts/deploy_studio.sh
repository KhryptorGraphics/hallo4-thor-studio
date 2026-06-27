#!/usr/bin/env bash
# Deploy Hallo4 Thor Studio as a persistent, secured systemd --user service.
#
# Idempotent: builds the frontend (if needed), generates a TLS cert + an auth
# token (if missing), installs/refreshes the systemd unit, and starts it. No
# sudo — runs as your user; with linger enabled it survives logout/reboot.
#
#   bash scripts/deploy_studio.sh            # deploy / refresh
#   systemctl --user status  hallo4-studio   # check
#   systemctl --user restart hallo4-studio   # restart (e.g. after a code change)
#   systemctl --user stop    hallo4-studio   # stop
#   systemctl --user disable --now hallo4-studio   # undeploy
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_NAME="${HALLO4_CONDA_ENV:-hallo4-thor}"
PORT="${HALLO4_STUDIO_PORT:-8443}"
CONDA="${HALLO4_CONDA_BIN:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}"
DATA="${HALLO4_STUDIO_DATA:-$REPO/studio_data}"
CERT_DIR="$DATA/certs"
ENV_FILE="$DATA/studio.env"
UNIT="hallo4-studio.service"
UNIT_DIR="$HOME/.config/systemd/user"

echo "==> Hallo4 Thor Studio deploy (env=$ENV_NAME port=$PORT)"

# 1. Frontend (the backend serves studio/frontend/dist).
if [ ! -f studio/frontend/dist/index.html ]; then
  echo "==> building frontend"
  ( cd studio/frontend && npm install && npm run build )
fi

# 2. Self-signed TLS cert (browser camera/mic need a secure context over the LAN).
if [ ! -f "$CERT_DIR/studio.key" ]; then
  echo "==> generating TLS cert"
  bash scripts/make_studio_cert.sh
fi

# 3. Auth — OFF by default (this is meant for a trusted private LAN). Opt in with
#    HALLO4_STUDIO_AUTH=1 to require a bearer token (e.g. if you expose it wider).
mkdir -p "$DATA"
if [ "${HALLO4_STUDIO_AUTH:-0}" = "1" ]; then
  if ! grep -q '^HALLO4_STUDIO_TOKEN=' "$ENV_FILE" 2>/dev/null; then
    echo "==> auth ON — generating token -> $ENV_FILE"
    umask 077
    echo "HALLO4_STUDIO_TOKEN=$(openssl rand -hex 24)" > "$ENV_FILE"
  fi
  chmod 600 "$ENV_FILE"
else
  : > "$ENV_FILE"   # empty -> no token -> open access on the trusted LAN
fi

# 4. systemd --user unit. conda run gives the full CUDA env (LD_LIBRARY_PATH etc.).
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/$UNIT" <<EOF
[Unit]
Description=Hallo4 Thor Studio
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
Environment=HALLO4_LIVE_ENGINE=1
Environment=HALLO4_ATTENTION_BACKEND=sdpa
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-$ENV_FILE
ExecStart=$CONDA run --no-capture-output -n $ENV_NAME uvicorn studio.backend.hallo4_studio.app:app \\
  --host 0.0.0.0 --port $PORT \\
  --ssl-keyfile $CERT_DIR/studio.key --ssl-certfile $CERT_DIR/studio.crt
# Cap memory so a leak OOM-kills+restarts only this service, never the shared box.
MemoryHigh=${HALLO4_STUDIO_MEM_HIGH:-24G}
MemoryMax=${HALLO4_STUDIO_MEM_MAX:-32G}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 5. (Re)start now. Boot autostart is OFF by default — opt in with HALLO4_STUDIO_BOOT=1.
systemctl --user daemon-reload
if [ "${HALLO4_STUDIO_BOOT:-0}" = "1" ]; then
  systemctl --user enable "$UNIT"
else
  systemctl --user disable "$UNIT" 2>/dev/null || true
fi
systemctl --user restart "$UNIT"   # start now + pick up code/cert changes
sleep 4

LAN="$(hostname -I | awk '{print $1}')"
TOKEN="$(. "$ENV_FILE" 2>/dev/null; echo "${HALLO4_STUDIO_TOKEN:-}")"
echo
systemctl --user --no-pager status "$UNIT" | head -6 || true
echo
if [ -n "$TOKEN" ]; then
  echo "==> Studio:  https://$LAN:$PORT/?token=$TOKEN   (auth ON)"
else
  echo "==> Studio:  https://$LAN:$PORT/   (no auth — trusted LAN)"
fi
echo "==> Logs:    journalctl --user -u $UNIT -f"
