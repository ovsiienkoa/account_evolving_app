#!/bin/bash

# Load environment variables from .env if it exists
if [ -f .env ]; then
  # export all variables from .env
  set -a
  source .env
  set +a
fi

mkdir -p .streamlit

cat > .streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${GOOGLE_REDIRECT_URI:-http://localhost:8501/oauth2callback}"
cookie_secret = "${COOKIE_SECRET:-$(openssl rand -hex 16 2>/dev/null || echo 'default_secret_please_change')}"

[auth.google]
client_id = "${GOOGLE_CLIENT_ID:-YOUR_GOOGLE_CLIENT_ID}"
client_secret = "${GOOGLE_CLIENT_SECRET:-YOUR_GOOGLE_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF

echo ".streamlit/secrets.toml has been created/updated."
echo "If you need to update it again later, just run this script with the environment variables set."
