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

# 3. Auth token — required: this animates likenesses/voices; never expose it open.
mkdir -p "$DATA"
if [ ! -f "$ENV_FILE" ]; then
  echo "==> generating auth token -> $ENV_FILE"
  umask 077
  echo "HALLO4_STUDIO_TOKEN=$(openssl rand -hex 24)" > "$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

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
EnvironmentFile=$ENV_FILE
ExecStart=$CONDA run --no-capture-output -n $ENV_NAME uvicorn studio.backend.hallo4_studio.app:app \\
  --host 0.0.0.0 --port $PORT \\
  --ssl-keyfile $CERT_DIR/studio.key --ssl-certfile $CERT_DIR/studio.crt
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 5. Enable + (re)start.
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT"
systemctl --user restart "$UNIT"   # pick up code/cert changes on re-deploy
sleep 4

LAN="$(hostname -I | awk '{print $1}')"
TOKEN="$(. "$ENV_FILE"; echo "$HALLO4_STUDIO_TOKEN")"
echo
systemctl --user --no-pager status "$UNIT" | head -6 || true
echo
echo "==> Studio:  https://$LAN:$PORT/"
echo "==> Token:   $TOKEN   (enter under Preflight > Access in the UI)"
echo "==> Logs:    journalctl --user -u $UNIT -f"
