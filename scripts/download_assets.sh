#!/usr/bin/env bash
set -e

# Create weights folder
mkdir -p app/weights

echo "⬇️ Downloading vox-256.yaml..."
wget -O app/weights/vox-256.yaml \
  https://raw.githubusercontent.com/AliaksandrSiarohin/first-order-model/master/config/vox-256.yaml

echo "⬇️ Downloading vox-cpk.pth.tar from Google Drive..."
FILE_ID="1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS"

# Get Google Drive confirm token and download
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
    -O app/weights/vox-cpk.pth.tar

rm -rf /tmp/cookies.txt

echo "✅ Weights downloaded successfully."
